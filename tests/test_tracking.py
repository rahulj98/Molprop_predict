"""Tests for experiment tracking.

Every test points MLflow at its own ``tmp_path`` store, so nothing here touches
the real ``mlruns/`` directory and no test can see another's runs. The networks
trained are deliberately tiny -- what is under test is the record-keeping, not
the learning.
"""

from __future__ import annotations

import json

import mlflow
import numpy as np
import pandas as pd
import pytest

from molecular_property_predictor.tracking import (
    best_run,
    config_params,
    configure,
    load_run_artifact,
    make_grid,
    name_config,
    search_runs,
    sweep,
    track_training,
    tracking_uri,
)
from molecular_property_predictor.train import TrainingConfig


@pytest.fixture
def frame() -> pd.DataFrame:
    """200 fake molecules whose target is a learnable function of composition."""
    generator = np.random.default_rng(0)
    molecules, targets = [], []
    for _ in range(200):
        n_carbon = int(generator.integers(1, 6))
        n_oxygen = int(generator.integers(0, 4))
        molecules.append([6] * n_carbon + [8] * n_oxygen + [1, 1])
        targets.append(1.5 * n_carbon - 0.8 * n_oxygen + generator.normal(scale=0.05))

    return pd.DataFrame(
        {
            "index": np.arange(1, 201),
            "lumo": targets,
            "atomic_numbers": molecules,
            "coordinates": [[0.0, 0.0, 0.0] * len(m) for m in molecules],
        }
    )


@pytest.fixture
def config() -> TrainingConfig:
    return TrainingConfig(
        representation="composition",
        hidden_sizes=(8, 4),
        dropout=0.0,
        batch_size=32,
        max_epochs=4,
        patience=4,
    )


@pytest.fixture
def store(tmp_path):
    """An isolated tracking store, plus the directory to pass everywhere."""
    directory = tmp_path / "mlruns"
    configure(directory, experiment="test-experiment")
    return directory


# --- Configuration ----------------------------------------------------------


def test_tracking_uri_is_absolute(tmp_path):
    """A relative URI would give a notebook and a script two separate stores --
    the same bug already fixed once for the data directory in Phase 2."""
    uri = tracking_uri(tmp_path / "mlruns")

    assert uri.startswith("sqlite:///")
    assert uri.endswith("mlruns/mlflow.db")
    assert "\\" not in uri  # SQLAlchemy wants forward slashes, Windows included


def test_configure_creates_the_experiment(tmp_path):
    experiment_id = configure(tmp_path / "mlruns", experiment="fresh")

    assert experiment_id
    assert (tmp_path / "mlruns" / "mlflow.db").exists()
    assert mlflow.get_experiment_by_name("fresh") is not None


def test_artifacts_land_beside_the_database_not_in_the_working_directory(tmp_path):
    """MLflow's default artifact root is relative to the cwd, which would drop
    checkpoints wherever a run happened to be launched from."""
    configure(tmp_path / "mlruns", experiment="located")
    experiment = mlflow.get_experiment_by_name("located")

    assert experiment.artifact_location.startswith("file://")
    assert experiment.artifact_location.endswith("mlruns/artifacts")


# --- Params -----------------------------------------------------------------


def test_config_params_covers_every_field(config):
    params = config_params(config)

    assert set(params) == {
        "representation", "hidden_sizes", "dropout", "loss_name", "learning_rate",
        "batch_size", "max_epochs", "patience", "split_seed", "torch_seed",
    }
    assert all(isinstance(value, str) for value in params.values())
    assert params["hidden_sizes"] == "(8, 4)"


# --- One tracked run --------------------------------------------------------


def test_track_training_logs_params_and_summary_metrics(frame, config, store, tmp_path):
    run_id, run = track_training(
        frame, config, run_name="one",
        processed_dir=tmp_path, model_dir=tmp_path / "models", device="cpu",
    )

    logged = mlflow.get_run(run_id)

    assert logged.data.params["learning_rate"] == str(config.learning_rate)
    assert logged.data.params["representation"] == "composition"
    assert logged.data.params["n_features"] == "5"
    assert logged.data.params["n_train"] == "160"  # 80% of 200
    assert logged.data.metrics["best_val_mae_ev"] == pytest.approx(
        run.metrics["mae_ev"], rel=1e-6
    )
    assert logged.data.metrics["best_epoch"] == run.best_epoch
    assert logged.data.tags["target"] == "lumo"


def test_metrics_are_logged_once_per_epoch(frame, config, store, tmp_path):
    """The per-epoch series is the difference between a tracked run and a row
    in a table: two runs can end at the same MAE by very different routes."""
    run_id, run = track_training(
        frame, config, run_name="curve",
        processed_dir=tmp_path, model_dir=tmp_path / "models", device="cpu",
    )

    client = mlflow.tracking.MlflowClient()
    history = client.get_metric_history(run_id, "mae_ev")

    assert len(history) == len(run.history)
    assert [point.step for point in history] == list(range(1, len(run.history) + 1))
    assert history[-1].value == pytest.approx(run.history["mae_ev"].iloc[-1], rel=1e-6)


def test_train_loss_is_also_stepped(frame, config, store, tmp_path):
    run_id, run = track_training(
        frame, config, run_name="loss",
        processed_dir=tmp_path, model_dir=tmp_path / "models", device="cpu",
    )

    client = mlflow.tracking.MlflowClient()
    history = client.get_metric_history(run_id, "train_loss")

    assert len(history) == len(run.history)


def test_checkpoint_and_history_are_attached_to_the_run(frame, config, store, tmp_path):
    run_id, _ = track_training(
        frame, config, run_name="files",
        processed_dir=tmp_path, model_dir=tmp_path / "models", device="cpu",
    )

    client = mlflow.tracking.MlflowClient()
    paths = {item.path for item in client.list_artifacts(run_id)}

    assert any(path.endswith(".pt") for path in paths)
    assert any(path.startswith("history_") for path in paths)
    assert "signature.json" in paths


def test_signature_records_the_input_shape(frame, config, store, tmp_path):
    """Phase 6 has to validate request payloads against something. Recording
    the contract with the run means it is not folklore."""
    run_id, _ = track_training(
        frame, config, run_name="sig",
        processed_dir=tmp_path, model_dir=tmp_path / "models", device="cpu",
    )

    local = mlflow.artifacts.download_artifacts(
        run_id=run_id, artifact_path="signature.json"
    )
    signature = json.loads(open(local).read())

    assert "inputs" in signature
    assert "5" in signature["inputs"]  # the five composition features


# --- Getting the model back out ---------------------------------------------


def test_load_run_artifact_reproduces_the_predictions(frame, config, store, tmp_path):
    """The point of tracking the checkpoint rather than just its score: a row
    in a results table leads back to the exact weights behind it."""
    from molecular_property_predictor.model import predict

    run_id, run = track_training(
        frame, config, run_name="roundtrip",
        processed_dir=tmp_path, model_dir=tmp_path / "models", device="cpu",
    )
    model, scaler, metadata = load_run_artifact(run_id, directory=store, device="cpu")

    features = np.array([[4.0, 3.0, 0.0, 1.0, 0.0], [2.0, 1.0, 0.0, 0.0, 0.0]])
    reloaded = predict(model, scaler, features)
    original = predict(run.model, run.scaler, features, device="cpu")

    np.testing.assert_allclose(reloaded, original, rtol=1e-6)
    assert metadata.representation == "composition"


def test_load_run_artifact_rejects_a_run_with_no_checkpoint(store):
    with mlflow.start_run() as active:
        mlflow.log_param("nothing", "attached")
        run_id = active.info.run_id

    with pytest.raises(LookupError, match="no .pt artifact"):
        load_run_artifact(run_id, directory=store)


# --- Grids ------------------------------------------------------------------


def test_make_grid_is_the_full_product(config):
    grid = make_grid(config, learning_rate=(1e-2, 1e-3), dropout=(0.0, 0.1, 0.2))

    assert len(grid) == 6
    assert {(c.learning_rate, c.dropout) for c in grid} == {
        (1e-2, 0.0), (1e-2, 0.1), (1e-2, 0.2),
        (1e-3, 0.0), (1e-3, 0.1), (1e-3, 0.2),
    }


def test_make_grid_leaves_other_fields_alone(config):
    grid = make_grid(config, learning_rate=(1e-2, 1e-3))

    assert all(c.representation == config.representation for c in grid)
    assert all(c.batch_size == config.batch_size for c in grid)


def test_make_grid_rejects_unknown_fields_before_training_anything(config):
    """Failing on the typo is worth more than failing after the first fit."""
    with pytest.raises(TypeError, match="learnig_rate"):
        make_grid(config, learnig_rate=(1e-2,))


def test_name_config_describes_what_varies(config):
    from dataclasses import replace

    name = name_config(replace(config, learning_rate=0.01), ["learning_rate"])
    widths = name_config(config, ["hidden_sizes"])

    assert name == "le0.01"
    assert widths == "hi8x4"


# --- Sweeps -----------------------------------------------------------------


def test_sweep_records_one_run_per_config(frame, config, store, tmp_path):
    grid = make_grid(config, learning_rate=(1e-2, 1e-3))

    results = sweep(
        frame, grid, varied=["learning_rate"], sweep_name="lr",
        processed_dir=tmp_path, model_dir=tmp_path / "models", device="cpu",
    )

    assert len(results) == 2
    assert set(results["run_id"]).__len__() == 2
    assert list(results["val_mae_ev"]) == sorted(results["val_mae_ev"])
    assert set(results["learning_rate"]) == {1e-2, 1e-3}


def test_sweep_runs_are_tagged_so_the_group_can_be_recovered(
    frame, config, store, tmp_path
):
    sweep(
        frame, make_grid(config, learning_rate=(1e-2,)), varied=["learning_rate"],
        sweep_name="tagged",
        processed_dir=tmp_path, model_dir=tmp_path / "models", device="cpu",
    )

    found = search_runs(
        experiment="test-experiment", directory=store,
        filter_string="tags.sweep = 'tagged'",
    )

    assert len(found) == 1


# --- Querying ---------------------------------------------------------------


def test_search_runs_returns_empty_for_an_unknown_experiment(store):
    """A fresh clone with no runs is a normal state, not an error."""
    assert search_runs(experiment="never-created", directory=store).empty


def test_search_runs_filters_on_params(frame, config, store, tmp_path):
    from dataclasses import replace

    track_training(
        frame, config, run_name="mse-run",
        processed_dir=tmp_path, model_dir=tmp_path / "models", device="cpu",
    )
    track_training(
        frame, replace(config, loss_name="l1"), run_name="l1-run",
        processed_dir=tmp_path, model_dir=tmp_path / "models", device="cpu",
    )

    found = search_runs(
        experiment="test-experiment", directory=store,
        filter_string="params.loss_name = 'l1'",
    )

    assert len(found) == 1
    assert found.iloc[0]["tags.mlflow.runName"] == "l1-run"


def test_search_runs_survives_a_new_client(frame, config, store, tmp_path):
    """The record has to outlive the process that wrote it -- that is the whole
    complaint against keeping results in a notebook kernel."""
    track_training(
        frame, config, run_name="persisted",
        processed_dir=tmp_path, model_dir=tmp_path / "models", device="cpu",
    )

    mlflow.set_tracking_uri("file:///nonexistent-elsewhere")
    found = search_runs(experiment="test-experiment", directory=store)

    assert len(found) == 1
    assert found.iloc[0]["tags.mlflow.runName"] == "persisted"


def test_best_run_picks_the_lowest_validation_mae(frame, config, store, tmp_path):
    from dataclasses import replace

    sweep(
        frame,
        [replace(config, learning_rate=1e-1), replace(config, learning_rate=1e-2)],
        varied=["learning_rate"], sweep_name="pick",
        processed_dir=tmp_path, model_dir=tmp_path / "models", device="cpu",
    )

    runs = search_runs(experiment="test-experiment", directory=store)
    best = best_run(experiment="test-experiment", directory=store)

    assert best["metrics.best_val_mae_ev"] == runs["metrics.best_val_mae_ev"].min()


def test_best_run_raises_when_there_is_nothing_to_pick(store):
    with pytest.raises(LookupError):
        best_run(experiment="empty-experiment", directory=store)
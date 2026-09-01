"""Tests for the training loop.

Run against a small synthetic dataset with a real signal in it, so the loop has
something to actually learn. No QM9, no feature cache -- these have to stay
fast enough to run on every change.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
import torch

from molecular_property_predictor.model import predict
from molecular_property_predictor.train import (
    LOSSES,
    TrainingConfig,
    evaluate_model,
    evaluate_on_test,
    prepare_data,
    record_test_score,
    set_seed,
    shuffled_batches,
    train_model,
)


@pytest.fixture
def frame() -> pd.DataFrame:
    """400 fake molecules whose target is a learnable function of composition.

    The composition representation is used because it is computed from the
    atomic numbers alone, so no coordinates or feature cache are needed.
    """
    generator = np.random.default_rng(0)
    molecules, targets = [], []
    for _ in range(400):
        n_carbon = int(generator.integers(1, 6))
        n_oxygen = int(generator.integers(0, 4))
        atomic_numbers = [6] * n_carbon + [8] * n_oxygen + [1, 1]
        molecules.append(atomic_numbers)
        targets.append(1.5 * n_carbon - 0.8 * n_oxygen + generator.normal(scale=0.05))

    return pd.DataFrame(
        {
            "index": np.arange(1, 401),
            "lumo": targets,
            "atomic_numbers": molecules,
            "coordinates": [[0.0, 0.0, 0.0] * len(m) for m in molecules],
        }
    )


@pytest.fixture
def config() -> TrainingConfig:
    return TrainingConfig(
        representation="composition",
        hidden_sizes=(32, 16),
        dropout=0.0,
        batch_size=32,
        max_epochs=40,
        patience=40,  # no early stop unless a test asks for it
    )


# --- Data preparation -------------------------------------------------------


def test_scaler_is_fitted_on_training_rows_only(frame, config, tmp_path):
    """The Phase 3 leakage guard, carried forward.

    Here the scaling is done by hand rather than by a scikit-learn Pipeline,
    so nothing structural prevents the mistake. Only this assertion does.
    """
    x_train, _, x_validation, _, scaler = prepare_data(
        frame, config, processed_dir=tmp_path
    )

    from molecular_property_predictor.baseline import split_positions
    from molecular_property_predictor.features import load_features

    train_rows, _, _ = split_positions(frame, seed=config.split_seed)
    features = load_features(frame, "composition", tmp_path)

    np.testing.assert_allclose(
        scaler.mean_, features[train_rows].mean(axis=0), rtol=1e-6
    )
    assert not np.allclose(scaler.mean_, features.mean(axis=0))
    # And the training rows really are standardised by those statistics.
    np.testing.assert_allclose(x_train.mean(axis=0), 0.0, atol=1e-5)
    assert len(x_validation) < len(x_train)


def test_prepare_data_splits_sizes_consistently(frame, config, tmp_path):
    x_train, y_train, x_validation, y_validation, _ = prepare_data(
        frame, config, processed_dir=tmp_path
    )

    assert len(x_train) == len(y_train) == 320  # 80% of 400
    assert len(x_validation) == len(y_validation) == 40  # 10%


# --- Batching ---------------------------------------------------------------
#
# `shuffled_batches` replaced a `DataLoader`, and the failure it can introduce
# is silent: an off-by-one in the slicing would drop rows from every epoch, the
# model would simply train on less data, and every other test here would still
# pass. So the contract is asserted directly rather than inferred from whether
# training still converges.


def test_shuffled_batches_visits_every_row_exactly_once():
    """The property a DataLoader gave us for free, and the one worth keeping."""
    seen = torch.cat(list(shuffled_batches(100, 16)))

    assert sorted(seen.tolist()) == list(range(100))


def test_shuffled_batches_includes_the_trailing_partial_batch():
    """100 rows in batches of 16 is six full batches and a remainder of four.

    Dropping the remainder is the classic way to lose data here: it is what
    ``DataLoader(drop_last=True)`` does on purpose, and it costs a silent 4% of
    every epoch if it happens by accident.
    """
    sizes = [len(rows) for rows in shuffled_batches(100, 16)]

    assert sizes == [16, 16, 16, 16, 16, 16, 4]


def test_shuffled_batches_divides_evenly_when_it_can():
    sizes = [len(rows) for rows in shuffled_batches(96, 16)]

    assert sizes == [16] * 6


def test_shuffled_batches_reshuffles_between_epochs():
    """A fixed order across epochs would make the shuffling decorative."""
    set_seed(0)
    first = torch.cat(list(shuffled_batches(100, 16)))
    second = torch.cat(list(shuffled_batches(100, 16)))

    assert not torch.equal(first, second)


def test_shuffled_batches_is_reproducible_for_a_given_seed():
    """Seeded runs must agree, or `torch_seed` in the metadata means nothing."""
    set_seed(0)
    first = torch.cat(list(shuffled_batches(100, 16)))
    set_seed(0)
    second = torch.cat(list(shuffled_batches(100, 16)))

    assert torch.equal(first, second)


def test_shuffled_batches_handles_a_batch_larger_than_the_data():
    sizes = [len(rows) for rows in shuffled_batches(10, 256)]

    assert sizes == [10]


# --- The loop ---------------------------------------------------------------


def test_training_learns_something(frame, config, tmp_path):
    """The target is a linear function of the features plus small noise, so a
    network that cannot beat the mean baseline is broken."""
    run = train_model(frame, config, processed_dir=tmp_path, device="cpu")

    _, y_train, _, y_validation, _ = prepare_data(frame, config, processed_dir=tmp_path)
    mean_baseline = np.abs(y_validation - y_train.mean()).mean()

    assert run.metrics["mae_ev"] < mean_baseline / 2
    assert run.metrics["r2"] > 0.9


def test_training_loss_decreases(frame, config, tmp_path):
    run = train_model(frame, config, processed_dir=tmp_path, device="cpu")

    losses = run.history["train_loss"].to_numpy()
    assert losses[-1] < losses[0]


def test_history_has_one_row_per_epoch(frame, config, tmp_path):
    run = train_model(frame, config, processed_dir=tmp_path, device="cpu")

    assert list(run.history["epoch"]) == list(range(1, len(run.history) + 1))
    assert set(run.history.columns) >= {
        "epoch", "train_loss", "mae_ev", "rmse_ev", "r2"
    }


def test_returned_model_carries_the_best_epoch_not_the_last(frame, config, tmp_path):
    """Validation MAE bounces around; the final epoch is often not the best.

    The reported metric must match the weights actually returned.
    """
    run = train_model(frame, config, processed_dir=tmp_path, device="cpu")

    best_row = run.history.loc[run.history["epoch"] == run.best_epoch].iloc[0]
    assert run.metrics["mae_ev"] == pytest.approx(best_row["mae_ev"], rel=1e-5)
    assert run.metrics["mae_ev"] <= run.history["mae_ev"].min() + 1e-9


def test_early_stopping_halts_before_max_epochs(frame, tmp_path):
    """With patience of 1 on an easy problem, training must stop early."""
    config = TrainingConfig(
        representation="composition",
        hidden_sizes=(32, 16),
        dropout=0.0,
        batch_size=32,
        max_epochs=200,
        patience=1,
    )

    run = train_model(frame, config, processed_dir=tmp_path, device="cpu")

    assert len(run.history) < 200
    assert run.best_epoch >= len(run.history) - 1


def test_training_is_reproducible_for_a_given_seed(frame, config, tmp_path):
    first = train_model(frame, config, processed_dir=tmp_path, device="cpu")
    second = train_model(frame, config, processed_dir=tmp_path, device="cpu")

    assert first.metrics["mae_ev"] == pytest.approx(second.metrics["mae_ev"])
    pd.testing.assert_frame_equal(first.history, second.history)


def test_a_different_torch_seed_gives_a_different_run(frame, config, tmp_path):
    """Otherwise the seed is decorative and reproducibility means nothing."""
    from dataclasses import replace

    first = train_model(frame, config, processed_dir=tmp_path, device="cpu")
    second = train_model(
        frame, replace(config, torch_seed=99), processed_dir=tmp_path, device="cpu"
    )

    assert first.history["train_loss"].iloc[0] != second.history["train_loss"].iloc[0]


@pytest.mark.parametrize("loss_name", sorted(LOSSES))
def test_every_loss_function_trains(frame, config, loss_name, tmp_path):
    from dataclasses import replace

    run = train_model(
        frame, replace(config, loss_name=loss_name, max_epochs=15),
        processed_dir=tmp_path, device="cpu",
    )

    assert run.metrics["r2"] > 0.5


def test_unknown_loss_is_rejected(frame, config, tmp_path):
    from dataclasses import replace

    with pytest.raises(ValueError, match="loss"):
        train_model(
            frame, replace(config, loss_name="crossentropy"),
            processed_dir=tmp_path, device="cpu",
        )


# --- The epoch callback -----------------------------------------------------
#
# The seam Phase 5 hangs experiment tracking on. Tested here rather than in
# test_tracking.py because the contract belongs to the loop: anything logging
# from it depends on being called once per epoch with these keys.


def test_on_epoch_is_called_once_per_epoch(frame, config, tmp_path):
    seen: list[tuple[int, dict]] = []

    run = train_model(
        frame, config, processed_dir=tmp_path, device="cpu",
        on_epoch=lambda epoch, metrics: seen.append((epoch, metrics)),
    )

    assert [epoch for epoch, _ in seen] == list(range(1, len(run.history) + 1))


def test_on_epoch_receives_the_loss_and_the_validation_metrics(
    frame, config, tmp_path
):
    seen: list[dict] = []

    run = train_model(
        frame, config, processed_dir=tmp_path, device="cpu",
        on_epoch=lambda epoch, metrics: seen.append(metrics),
    )

    assert set(seen[0]) == {"train_loss", "mae_ev", "rmse_ev", "r2"}
    assert seen[-1]["mae_ev"] == pytest.approx(run.history["mae_ev"].iloc[-1])
    assert all(isinstance(value, float) for value in seen[0].values())


def test_training_is_unchanged_by_the_callback(frame, config, tmp_path):
    """The callback observes; it must not be able to alter the run."""
    without = train_model(frame, config, processed_dir=tmp_path, device="cpu")
    with_callback = train_model(
        frame, config, processed_dir=tmp_path, device="cpu",
        on_epoch=lambda epoch, metrics: metrics.clear(),
    )

    pd.testing.assert_frame_equal(without.history, with_callback.history)


# --- Evaluation -------------------------------------------------------------


def test_evaluate_model_does_not_leave_the_model_in_train_mode(frame, config, tmp_path):
    run = train_model(frame, config, processed_dir=tmp_path, device="cpu")
    x = torch.randn(16, 5)

    evaluate_model(run.model, x, np.zeros(16))

    assert not run.model.training


def test_evaluate_model_is_unaffected_by_batching(frame, config, tmp_path):
    run = train_model(frame, config, processed_dir=tmp_path, device="cpu")
    x = torch.randn(37, 5)
    y = np.random.default_rng(0).normal(size=37)

    small = evaluate_model(run.model, x, y, batch_size=4)
    large = evaluate_model(run.model, x, y, batch_size=1024)

    assert small["mae_ev"] == pytest.approx(large["mae_ev"], rel=1e-5)


def test_evaluate_model_records_no_gradients(frame, config, tmp_path):
    """Scoring must not build a graph; nothing will ever call backward on it."""
    run = train_model(frame, config, processed_dir=tmp_path, device="cpu")
    run.model.zero_grad(set_to_none=True)

    evaluate_model(run.model, torch.randn(16, 5), np.zeros(16))

    assert all(p.grad is None for p in run.model.parameters())


# --- Metadata ---------------------------------------------------------------


def test_run_metadata_describes_the_run(frame, config, tmp_path):
    run = train_model(frame, config, processed_dir=tmp_path, device="cpu")
    metadata = run.metadata()

    assert metadata.representation == "composition"
    assert metadata.target == "lumo"
    assert metadata.n_features == 5
    assert metadata.best_epoch == run.best_epoch
    assert metadata.epochs_trained == len(run.history)
    assert metadata.validation_mae_ev == pytest.approx(run.metrics["mae_ev"])


# --- Caching ----------------------------------------------------------------


def test_train_and_cache_writes_an_artifact_and_a_history(frame, config, tmp_path):
    from molecular_property_predictor.train import train_and_cache

    artifact, history = train_and_cache(
        frame, config, "test", model_dir=tmp_path / "models", processed_dir=tmp_path
    )

    assert artifact.exists()
    assert (tmp_path / "models" / "history_test.csv").exists()
    assert len(history) > 0


def test_train_and_cache_reuses_the_cache(frame, config, tmp_path):
    """Proven by tampering: if the second call retrained, it would overwrite
    the edit rather than return it."""
    from molecular_property_predictor.train import train_and_cache

    model_dir = tmp_path / "models"
    train_and_cache(frame, config, "test", model_dir=model_dir, processed_dir=tmp_path)

    tampered = pd.read_csv(model_dir / "history_test.csv")
    tampered["mae_ev"] = -1.0
    tampered.to_csv(model_dir / "history_test.csv", index=False)

    _, history = train_and_cache(
        frame, config, "test", model_dir=model_dir, processed_dir=tmp_path
    )

    assert (history["mae_ev"] == -1.0).all()


def test_train_and_cache_can_be_forced_to_retrain(frame, config, tmp_path):
    from molecular_property_predictor.train import train_and_cache

    model_dir = tmp_path / "models"
    train_and_cache(frame, config, "test", model_dir=model_dir, processed_dir=tmp_path)

    tampered = pd.read_csv(model_dir / "history_test.csv")
    tampered["mae_ev"] = -1.0
    tampered.to_csv(model_dir / "history_test.csv", index=False)

    _, history = train_and_cache(
        frame, config, "test", model_dir=model_dir, processed_dir=tmp_path,
        use_cache=False,
    )

    assert (history["mae_ev"] > 0).all()


def test_cached_artifact_loads_and_predicts(frame, config, tmp_path):
    """The cache has to hold a usable model, not just a file of the right name."""
    from molecular_property_predictor.model import load_artifact, predict
    from molecular_property_predictor.train import train_and_cache

    artifact, _ = train_and_cache(
        frame, config, "test", model_dir=tmp_path / "models", processed_dir=tmp_path
    )
    model, scaler, metadata = load_artifact(artifact)

    predictions = predict(model, scaler, np.array([[4.0, 3.0, 0.0, 1.0, 0.0]]))

    assert predictions.shape == (1,)
    assert np.isfinite(predictions).all()
    assert metadata.representation == "composition"


def test_set_seed_makes_initial_weights_repeatable():
    from molecular_property_predictor.model import MolecularNet

    set_seed(11)
    first = MolecularNet(n_features=8, hidden_sizes=(4,)).layers[0].weight.clone()
    set_seed(11)
    second = MolecularNet(n_features=8, hidden_sizes=(4,)).layers[0].weight

    torch.testing.assert_close(first, second)


# --- The one-time test-set evaluation ---------------------------------------


@pytest.fixture
def saved_artifact(frame, config, tmp_path):
    """A trained checkpoint on disk, as `evaluate_on_test` expects one."""
    from molecular_property_predictor.model import save_artifact

    run = train_model(frame, config, processed_dir=tmp_path)
    return save_artifact(
        tmp_path / "trained.pt", run.model, run.scaler, run.metadata()
    )


def test_evaluate_on_test_returns_the_three_metrics(frame, saved_artifact, tmp_path):
    scores = evaluate_on_test(frame, saved_artifact, processed_dir=tmp_path)

    assert set(scores) == {"mae_ev", "rmse_ev", "r2"}
    assert all(np.isfinite(value) for value in scores.values())


def test_evaluate_on_test_scores_the_split_the_model_never_saw(
    frame, saved_artifact, tmp_path
):
    """The test rows must come from the artifact's own seed, not a default.

    Scoring a model against a split it was partly trained on does not raise --
    it just reports a number that is better than the truth. Two artifacts that
    differ only in `split_seed` must therefore be scored on different molecules,
    which is what this asserts.
    """
    from dataclasses import replace as replace_dataclass

    from molecular_property_predictor.model import load_artifact, save_artifact

    model, scaler, metadata = load_artifact(saved_artifact)
    other = save_artifact(
        tmp_path / "seed7.pt",
        model,
        scaler,
        replace_dataclass(metadata, split_seed=7),
    )

    with_seed_zero = evaluate_on_test(frame, saved_artifact, processed_dir=tmp_path)
    with_seed_seven = evaluate_on_test(frame, other, processed_dir=tmp_path)

    assert with_seed_zero["mae_ev"] != with_seed_seven["mae_ev"]


def test_record_test_score_stores_the_number_in_the_metadata(
    saved_artifact, tmp_path
):
    from molecular_property_predictor.model import load_artifact

    record_test_score(saved_artifact, 0.1234, destination=tmp_path / "served.pt")
    _, _, metadata = load_artifact(tmp_path / "served.pt")

    assert metadata.test_mae_ev == pytest.approx(0.1234)


def test_record_test_score_leaves_the_weights_untouched(
    frame, saved_artifact, tmp_path
):
    """Recording a score must not perturb the model that earned it."""
    from molecular_property_predictor.model import load_artifact

    features = np.array([[4.0, 3.0, 0.0, 1.0, 0.0]])
    before = predict(*load_artifact(saved_artifact)[:2], features)

    served = record_test_score(saved_artifact, 0.5, destination=tmp_path / "s.pt")
    after = predict(*load_artifact(served)[:2], features)

    np.testing.assert_allclose(before, after, rtol=0, atol=0)


def test_record_test_score_defaults_to_overwriting_in_place(saved_artifact):
    from molecular_property_predictor.model import load_artifact

    written = record_test_score(saved_artifact, 0.9)

    assert written == saved_artifact
    assert load_artifact(saved_artifact)[2].test_mae_ev == pytest.approx(0.9)


def test_artifacts_without_a_test_score_load_as_none(saved_artifact):
    """Backward compatibility: every checkpoint written before Phase 6."""
    from molecular_property_predictor.model import load_artifact

    assert load_artifact(saved_artifact)[2].test_mae_ev is None

"""Tests for the baseline models.

The important one here is
:func:`test_pipeline_fits_the_scaler_on_training_data_only`. Leakage through a
prematurely fitted scaler raises nothing, fails nothing, and simply makes every
reported score better than the truth. An assertion is the only thing that
catches it.

These run on small synthetic arrays -- no dataset, no feature cache.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from sklearn.dummy import DummyRegressor
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler

from molecular_property_predictor.baseline import (
    BASELINES,
    REPRESENTATIONS,
    evaluate,
    fit_and_select,
    make_pipeline,
    run_sweep,
    split_positions,
)
from molecular_property_predictor.data import split_dataset
from molecular_property_predictor.features import FEATURE_KINDS


@pytest.fixture
def regression_data() -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """A linear problem with wildly different feature scales, plus noise."""
    generator = np.random.default_rng(0)
    x = generator.normal(size=(400, 4)) * np.array([1.0, 1000.0, 0.01, 5.0])
    y = x @ np.array([2.0, 0.003, 50.0, -1.0]) + generator.normal(scale=0.1, size=400)
    return x[:300], y[:300], x[300:], y[300:]


# --- Metrics ----------------------------------------------------------------


def test_evaluate_computes_metrics_by_hand():
    y_true = np.array([1.0, 2.0, 3.0, 4.0])
    y_predicted = np.array([1.5, 2.0, 2.0, 4.0])  # errors: 0.5, 0, -1, 0

    metrics = evaluate(y_true, y_predicted)

    assert metrics["mae_ev"] == pytest.approx(1.5 / 4)
    assert metrics["rmse_ev"] == pytest.approx(np.sqrt(1.25 / 4))
    assert metrics["r2"] == pytest.approx(1 - 1.25 / 5.0)


def test_evaluate_scores_perfect_predictions():
    y_true = np.array([1.0, 2.0, 3.0])

    metrics = evaluate(y_true, y_true)

    assert metrics["mae_ev"] == 0.0
    assert metrics["rmse_ev"] == 0.0
    assert metrics["r2"] == pytest.approx(1.0)


def test_r2_of_the_mean_prediction_is_zero():
    """R^2 is measured against predicting the mean, so this must be exactly 0."""
    y_true = np.array([1.0, 5.0, 3.0, 9.0, 2.0])

    metrics = evaluate(y_true, np.full_like(y_true, y_true.mean()))

    assert metrics["r2"] == pytest.approx(0.0)


def test_rmse_separates_error_patterns_that_mae_cannot():
    """Documents what the pair of metrics is for.

    Both predictions below are wrong by the same *average* amount. Only RMSE
    distinguishes one catastrophic miss from a uniform small bias, which is why
    reporting MAE alone would hide the difference.
    """
    y_true = np.zeros(100)
    concentrated = np.zeros(100)
    concentrated[0] = 10.0  # one bad miss, ninety-nine perfect predictions
    spread = np.full(100, 0.1)  # every molecule slightly off

    concentrated_metrics = evaluate(y_true, concentrated)
    spread_metrics = evaluate(y_true, spread)

    assert concentrated_metrics["mae_ev"] == pytest.approx(spread_metrics["mae_ev"])
    assert concentrated_metrics["rmse_ev"] == pytest.approx(
        10 * spread_metrics["rmse_ev"]
    )


# --- Leakage ----------------------------------------------------------------


def test_pipeline_fits_the_scaler_on_training_data_only(regression_data):
    """The central guard of this module.

    If the scaler ever sees validation data during fit, its mean shifts towards
    the validation mean and every later score is quietly optimistic.
    """
    x_train, y_train, x_validation, _ = regression_data

    pipeline = make_pipeline(Ridge())
    pipeline.fit(x_train, y_train)
    scaler = pipeline.named_steps["scaler"]

    np.testing.assert_allclose(scaler.mean_, x_train.mean(axis=0))
    np.testing.assert_allclose(scaler.scale_, x_train.std(axis=0))

    everything = np.vstack([x_train, x_validation])
    assert not np.allclose(scaler.mean_, everything.mean(axis=0))


def test_predicting_does_not_refit_the_scaler(regression_data):
    """Validation must be transformed with training statistics, not its own."""
    x_train, y_train, x_validation, _ = regression_data

    pipeline = make_pipeline(Ridge())
    pipeline.fit(x_train, y_train)
    before = pipeline.named_steps["scaler"].mean_.copy()

    pipeline.predict(x_validation)

    np.testing.assert_array_equal(pipeline.named_steps["scaler"].mean_, before)


def test_pipeline_scaling_actually_standardises(regression_data):
    """Sanity: the scaler is doing something to these very uneven columns."""
    x_train, _, _, _ = regression_data

    scaled = StandardScaler().fit_transform(x_train)

    np.testing.assert_allclose(scaled.mean(axis=0), 0.0, atol=1e-12)
    np.testing.assert_allclose(scaled.std(axis=0), 1.0)


# --- Model selection --------------------------------------------------------


def test_fit_and_select_returns_the_best_alpha_on_validation(regression_data):
    x_train, y_train, x_validation, y_validation = regression_data
    grid = {"model__alpha": (0.01, 1.0, 1e6)}

    _, parameters, metrics = fit_and_select(
        Ridge(), grid, x_train, y_train, x_validation, y_validation
    )

    assert parameters["model__alpha"] in grid["model__alpha"]
    # A near-infinite penalty crushes the coefficients to nothing, so it can
    # only win if the search is broken.
    assert parameters["model__alpha"] != 1e6
    assert metrics["mae_ev"] < np.abs(y_validation - y_train.mean()).mean()


def test_fit_and_select_handles_an_empty_grid(regression_data):
    """Models with nothing to tune still have to fit and score."""
    x_train, y_train, x_validation, y_validation = regression_data

    pipeline, parameters, metrics = fit_and_select(
        DummyRegressor(strategy="mean"), {}, x_train, y_train, x_validation,
        y_validation,
    )

    assert parameters == {}
    assert set(metrics) == {"mae_ev", "rmse_ev", "r2"}
    np.testing.assert_allclose(pipeline.predict(x_validation), y_train.mean())


def test_fit_and_select_is_reproducible(regression_data):
    x_train, y_train, x_validation, y_validation = regression_data
    grid = {"model__alpha": (0.1, 1.0, 10.0)}

    first = fit_and_select(Ridge(), grid, x_train, y_train, x_validation, y_validation)
    second = fit_and_select(Ridge(), grid, x_train, y_train, x_validation, y_validation)

    assert first[1] == second[1]
    assert first[2] == second[2]


def test_the_mean_baseline_predicts_the_training_mean(regression_data):
    """This is the floor the whole project is measured against."""
    x_train, y_train, x_validation, y_validation = regression_data

    pipeline = make_pipeline(DummyRegressor(strategy="mean"))
    pipeline.fit(x_train, y_train)
    metrics = evaluate(y_validation, pipeline.predict(x_validation))

    assert metrics["mae_ev"] == pytest.approx(
        np.abs(y_validation - y_train.mean()).mean()
    )


# --- Splitting --------------------------------------------------------------


@pytest.fixture
def frame() -> pd.DataFrame:
    generator = np.random.default_rng(1)
    return pd.DataFrame(
        {
            "index": np.arange(1, 601),
            "lumo": generator.normal(size=600),
            "atomic_numbers": [[6, 1, 1, 1, 1]] * 600,
            "coordinates": [[0.0, 0.0, 0.0] + [1.0] * 12] * 600,
        }
    )


def test_split_positions_partitions_every_row_exactly_once(frame):
    train, validation, test = split_positions(frame)

    combined = np.concatenate([train, validation, test])
    np.testing.assert_array_equal(np.sort(combined), np.arange(len(frame)))


def test_split_positions_agrees_with_split_dataset(frame):
    """The positions must select exactly the molecules split_dataset returns."""
    train_rows, validation_rows, test_rows = split_positions(frame, seed=7)
    train, validation, test = split_dataset(frame, seed=7)

    for rows, expected in (
        (train_rows, train),
        (validation_rows, validation),
        (test_rows, test),
    ):
        np.testing.assert_array_equal(
            frame["index"].to_numpy()[rows], expected["index"].to_numpy()
        )


def test_split_positions_is_seed_dependent(frame):
    assert not np.array_equal(
        split_positions(frame, seed=0)[0], split_positions(frame, seed=1)[0]
    )


# --- The sweep --------------------------------------------------------------


def test_run_sweep_produces_one_row_per_pair(frame, tmp_path):
    results = run_sweep(
        frame,
        representations=("composition",),
        models=("mean", "ridge"),
        processed_dir=tmp_path,
    )

    assert len(results) == 2
    assert set(results["model"]) == {"mean", "ridge"}
    assert (results["n_features"] == 5).all()
    assert results["mae_ev"].is_monotonic_increasing  # sorted best-first


def test_run_sweep_rejects_an_unknown_representation(frame, tmp_path):
    with pytest.raises(ValueError, match="representation"):
        run_sweep(frame, representations=("morgan",), processed_dir=tmp_path)


def test_run_sweep_rejects_an_unknown_model(frame, tmp_path):
    with pytest.raises(ValueError, match="model"):
        run_sweep(
            frame,
            representations=("composition",),
            models=("transformer",),
            processed_dir=tmp_path,
        )


def test_every_representation_in_the_sweep_can_be_built():
    """Guards against a name drifting out of step with features.FEATURE_KINDS."""
    assert set(REPRESENTATIONS) <= set(FEATURE_KINDS)


def test_every_baseline_builds_a_fresh_unfitted_estimator():
    """Factories, not shared instances -- a fitted estimator must never be reused."""
    for name, (build, _) in BASELINES.items():
        first, second = build(0), build(0)

        assert first is not second, name
        assert not hasattr(first, "n_features_in_"), name

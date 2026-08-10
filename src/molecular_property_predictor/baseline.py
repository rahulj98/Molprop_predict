"""Baseline regressors: the numbers every later model has to beat.

A baseline is not a formality. It does three jobs that no amount of model
sophistication does for you:

1. **It makes later numbers interpretable.** "0.6 eV MAE" means nothing alone.
   "0.6 eV against a 1.05 eV mean-prediction floor and a 0.7 eV ridge
   regression" means something.
2. **It catches pipeline bugs.** If an elaborate model cannot beat ridge
   regression, the fault is usually upstream -- in the features or the split --
   and it is far cheaper to find that here than in a training loop.
3. **It justifies complexity.** A neural network that ties a twenty-line
   scikit-learn pipeline has not earned its place, and reporting that honestly
   is worth more than reporting the number alone.

This module also answers the two questions Phase 2 deliberately left open: does
a geometry-based representation beat plain atom counts, and does the
435-feature sorted Coulomb matrix beat the 29-feature eigenspectrum?

**The test set is not touched here.** Everything below scores on validation.
See :func:`run_sweep`.
"""

from __future__ import annotations

import time
from collections.abc import Callable, Iterable, Sequence
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator
from sklearn.dummy import DummyRegressor
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.linear_model import Ridge
from sklearn.metrics import r2_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from molecular_property_predictor.data import (
    DEFAULT_PROCESSED_DIR,
    TARGET_PROPERTY,
    split_dataset,
)
from molecular_property_predictor.features import FEATURE_KINDS, load_features

#: Representations to compare, cheapest first.
REPRESENTATIONS: tuple[str, ...] = ("composition", "eigenspectrum", "sorted_coulomb")


def evaluate(y_true: np.ndarray, y_predicted: np.ndarray) -> dict[str, float]:
    """Score predictions three ways.

    ``mae_ev`` is the headline: it is in eV, so it is directly interpretable
    and directly comparable to the QM9 literature.

    ``rmse_ev`` squares the errors before averaging, so it punishes large
    misses far harder than MAE does. Reporting both is informative -- when RMSE
    is much larger than MAE, the error is concentrated in a few badly predicted
    molecules rather than spread evenly, which points at a different fix.

    ``r2`` is the fraction of variance explained: 0 means no better than
    predicting the mean, 1 means perfect. Unlike the other two it is unitless,
    which makes it comparable across targets but hides how big the errors are.
    """
    error = np.asarray(y_predicted, dtype=np.float64) - np.asarray(
        y_true, dtype=np.float64
    )
    return {
        "mae_ev": float(np.mean(np.abs(error))),
        "rmse_ev": float(np.sqrt(np.mean(error**2))),
        "r2": float(r2_score(y_true, y_predicted)),
    }


def make_pipeline(estimator: BaseEstimator) -> Pipeline:
    """Wrap an estimator so its preprocessing travels with it.

    The scaler goes *inside* the pipeline, and that is the whole point. Calling
    ``pipeline.fit(X_train, y_train)`` fits the scaler on training data alone;
    ``pipeline.predict(X_val)`` then applies those same training statistics to
    validation. Standardising the full array up front instead -- the obvious
    shortcut -- computes means and standard deviations that include the
    validation and test molecules, and those statistics leak straight into
    training. Nothing raises, no test fails, and the reported score is
    flattering and wrong.

    Scaling is a no-op for tree ensembles, which are indifferent to monotone
    rescaling. It is kept anyway so that every baseline is the same kind of
    object, which matters once Phase 6 has to load one and serve it.
    """
    return Pipeline([("scaler", StandardScaler()), ("model", estimator)])


#: Baselines to fit, each as a factory plus a small grid searched on
#: validation. Factories rather than instances so that every fit starts from a
#: fresh, unfitted estimator.
BASELINES: dict[str, tuple[Callable[[int], BaseEstimator], dict[str, Sequence]]] = {
    # Predicts the training mean for every molecule. Reproduces the floor from
    # Phase 1 as a computed number rather than a constant in a docstring -- if
    # it ever disagrees, something upstream has broken.
    "mean": (lambda seed: DummyRegressor(strategy="mean"), {}),
    # The honest "is this problem linear?" test. `alpha` penalises large
    # coefficients; with 435 correlated features some regularisation is
    # necessary, and the right amount is an empirical question.
    "ridge": (
        lambda seed: Ridge(),
        {"model__alpha": (0.01, 0.1, 1.0, 10.0, 100.0, 1000.0)},
    ),
    # Non-linear, and unlike a random forest it bins features into 255 buckets
    # up front, so 100k x 435 fits in memory and trains in minutes.
    "gradient_boosting": (
        lambda seed: HistGradientBoostingRegressor(random_state=seed),
        {},
    ),
}


def split_positions(
    frame: pd.DataFrame, *, seed: int = 0
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Row positions of the train / validation / test split.

    :func:`~molecular_property_predictor.data.split_dataset` returns molecules,
    but the feature matrices are plain arrays aligned to ``frame`` row order,
    so what we need here are *positions*. Splitting a frame of positions with
    the same seed and the same length reproduces exactly the same partition,
    which keeps one seeded split shared across every phase.
    """
    positions = pd.DataFrame(
        {"index": frame["index"].to_numpy(), "row": np.arange(len(frame))}
    )
    train, validation, test = split_dataset(positions, seed=seed)
    return (
        train["row"].to_numpy(),
        validation["row"].to_numpy(),
        test["row"].to_numpy(),
    )


def fit_and_select(
    estimator: BaseEstimator,
    param_grid: dict[str, Sequence],
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_validation: np.ndarray,
    y_validation: np.ndarray,
) -> tuple[Pipeline, dict, dict[str, float]]:
    """Fit over a small grid and keep whatever scores best on validation.

    A single held-out validation set rather than k-fold cross-validation:
    with ~13,000 validation molecules the MAE estimate is good to about
    +/-0.01 eV, far finer than the differences being compared, so k-fold would
    cost five times the compute to reduce noise that is already negligible.
    Cross-validation earns its keep on small datasets, not this one.

    Returns:
        ``(fitted_pipeline, chosen_parameters, validation_metrics)``.
    """
    names = list(param_grid)
    combinations: list[dict] = [{}]
    for name in names:
        combinations = [
            {**existing, name: value}
            for existing in combinations
            for value in param_grid[name]
        ]

    best: tuple[Pipeline, dict, dict[str, float]] | None = None
    for parameters in combinations:
        pipeline = make_pipeline(estimator)
        pipeline.set_params(**parameters)
        pipeline.fit(x_train, y_train)
        metrics = evaluate(y_validation, pipeline.predict(x_validation))
        if best is None or metrics["mae_ev"] < best[2]["mae_ev"]:
            best = (pipeline, parameters, metrics)

    assert best is not None  # combinations always holds at least {}
    return best


def run_sweep(
    frame: pd.DataFrame,
    representations: Iterable[str] = REPRESENTATIONS,
    models: Iterable[str] = tuple(BASELINES),
    *,
    target: str = TARGET_PROPERTY,
    seed: int = 0,
    processed_dir: Path | str = DEFAULT_PROCESSED_DIR,
    report: Callable[[str], None] | None = None,
) -> pd.DataFrame:
    """Fit every (representation, model) pair and score it on validation.

    Scores come from the **validation** split, never the test split. We are
    about to make choices by looking at these numbers -- which representation,
    which model, which regularisation strength -- and every choice made against
    a set spends part of that set's ability to estimate unseen performance. The
    test split is opened once, at the end of the project, on one final model.

    Returns:
        One row per pair, sorted by validation MAE.
    """
    unknown = set(representations) - set(FEATURE_KINDS)
    if unknown:
        raise ValueError(f"unknown representation(s): {sorted(unknown)}")
    unknown = set(models) - set(BASELINES)
    if unknown:
        raise ValueError(f"unknown model(s): {sorted(unknown)}")

    train_rows, validation_rows, _ = split_positions(frame, seed=seed)
    y = frame[target].to_numpy(dtype=np.float64)
    y_train, y_validation = y[train_rows], y[validation_rows]

    rows = []
    for representation in representations:
        features = load_features(frame, representation, processed_dir)
        x_train = features[train_rows]
        x_validation = features[validation_rows]

        for model in models:
            build, grid = BASELINES[model]
            start = time.perf_counter()
            _, parameters, metrics = fit_and_select(
                build(seed), grid, x_train, y_train, x_validation, y_validation
            )
            elapsed = time.perf_counter() - start

            rows.append(
                {
                    "representation": representation,
                    "model": model,
                    "n_features": features.shape[1],
                    **metrics,
                    "params": ", ".join(f"{k.split('__')[-1]}={v}"
                                        for k, v in parameters.items()) or "-",
                    "fit_seconds": round(elapsed, 1),
                }
            )
            if report is not None:
                report(
                    f"{representation:<15} {model:<18} "
                    f"MAE {metrics['mae_ev']:.4f} eV   "
                    f"R2 {metrics['r2']:.4f}   ({elapsed:.1f}s)"
                )

    return pd.DataFrame(rows).sort_values("mae_ev", ignore_index=True)


def load_sweep(
    frame: pd.DataFrame,
    *,
    seed: int = 0,
    processed_dir: Path | str = DEFAULT_PROCESSED_DIR,
    use_cache: bool = True,
    report: Callable[[str], None] | None = None,
) -> pd.DataFrame:
    """Run the full sweep, caching the results table to CSV.

    The gradient-boosting fits take minutes, and a notebook should not repeat
    them on every execution.
    """
    cache = Path(processed_dir) / f"baseline_results_seed{seed}.csv"
    if use_cache and cache.exists():
        return pd.read_csv(cache)

    results = run_sweep(
        frame, seed=seed, processed_dir=processed_dir, report=report
    )
    cache.parent.mkdir(parents=True, exist_ok=True)
    results.to_csv(cache, index=False)
    return results


def learning_curve(
    frame: pd.DataFrame,
    representation: str,
    model: str,
    training_sizes: Sequence[int],
    *,
    target: str = TARGET_PROPERTY,
    seed: int = 0,
    processed_dir: Path | str = DEFAULT_PROCESSED_DIR,
) -> pd.DataFrame:
    """Validation MAE as a function of how much training data the model sees.

    This answers a question worth settling before building anything larger: is
    performance limited by the amount of data or by the model? A curve still
    falling at the full training set size says more data would help. A curve
    that has flattened says the model or the representation is the ceiling, and
    Phase 4 should expect its gains to come from there.

    The subsets are nested prefixes of the (already shuffled) training split,
    so each larger set contains the smaller ones and the comparison is not
    confounded by which molecules happened to be drawn.
    """
    train_rows, validation_rows, _ = split_positions(frame, seed=seed)
    features = load_features(frame, representation, processed_dir)
    y = frame[target].to_numpy(dtype=np.float64)

    x_validation, y_validation = features[validation_rows], y[validation_rows]
    build, grid = BASELINES[model]

    rows = []
    for size in training_sizes:
        subset = train_rows[:size]
        _, _, metrics = fit_and_select(
            build(seed),
            grid,
            features[subset],
            y[subset],
            x_validation,
            y_validation,
        )
        rows.append({"n_train": size, **metrics})

    return pd.DataFrame(rows)

"""Regenerate the figures used in README.md.

Why this is a script and not a notebook cell: a picture in a README is a claim,
and a claim in a portfolio repository should be reproducible by whoever is
reading it. Anyone can run this and get exactly the plots the README shows,
from the same artifact the API serves.

Two figures, chosen because each answers a question a reader actually has.

``model_comparison.png`` answers "did the complicated thing beat the simple
thing?" -- the honest question to ask of any neural network, and one many
projects avoid by never fitting a baseline at all.

``predicted_vs_true.png`` answers "where does it fail?" It shows the R² = 0.92
and the shrinkage toward the mean in the same frame, which is the point: the
aggregate score and the systematic weakness are properties of one model, and
showing only the first is how honest projects turn into misleading ones.

Both use *validation* scores, because those are the numbers every modelling
choice was made against and therefore the only ones comparable across phases.
The single test-set figure is reported in the README as text, where it cannot
be mistaken for one of a series.

Run from the repository root::

    python scripts/make_figures.py

Requires the QM9 cache in data/processed/ and models/served.pt. Both are build
outputs; see the README for how to produce them.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

# Agg: render to a file, never to a window. Without this the script blocks on a
# machine with no display -- which is every CI runner and most servers.
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from molecular_property_predictor.data import (
    DEFAULT_PROCESSED_DIR,
    load_qm9,
    split_dataset,
)
from molecular_property_predictor.features import load_features
from molecular_property_predictor.model import DEFAULT_MODEL_DIR, load_artifact, predict

FIGURE_DIR = Path(__file__).resolve().parent.parent / "docs" / "figures"

#: Recorded validation scores from Phase 3, written by notebook 03.
BASELINE_RESULTS = DEFAULT_PROCESSED_DIR / "baseline_results_seed0.csv"

#: The checkpoint the API serves, and the source of the network's own score.
SERVED_ARTIFACT = DEFAULT_MODEL_DIR / "served.pt"

# A restrained palette: one colour for the baselines, one for the model this
# project is about. Colour carries meaning here rather than decoration -- the
# reader should see at a glance which bar is the thing being argued for.
BASELINE_COLOUR = "#9aa6b2"
MODEL_COLOUR = "#1f4e79"
GRID_COLOUR = "#dde3e8"


def style_axes(ax) -> None:
    """Strip the chartjunk matplotlib adds by default."""
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    ax.grid(axis="both", color=GRID_COLOUR, linewidth=0.8, zorder=0)
    ax.set_axisbelow(True)


def figure_model_comparison(validation_mae: float) -> Path:
    """Bar chart of validation MAE, simplest model to most complex.

    Args:
        validation_mae: The served network's score, read from its own metadata
            rather than hardcoded, so the figure cannot drift from the artifact.
    """
    if not BASELINE_RESULTS.exists():
        raise FileNotFoundError(
            f"{BASELINE_RESULTS} not found. Run notebooks/03_baseline.ipynb to "
            "produce it."
        )

    results = pd.read_csv(BASELINE_RESULTS)

    # The best score per model family, so the chart compares approaches rather
    # than every representation crossed with every model.
    rows = [
        ("Predict the mean", float(results.query("model == 'mean'")["mae_ev"].iloc[0])),
        (
            "Ridge\n(atom counts)",
            float(
                results.query("model == 'ridge' and representation == 'composition'")[
                    "mae_ev"
                ].iloc[0]
            ),
        ),
        (
            "Ridge\n(geometry)",
            float(
                results.query(
                    "model == 'ridge' and representation == 'sorted_coulomb'"
                )["mae_ev"].iloc[0]
            ),
        ),
        (
            "Gradient boosting\n(geometry)",
            float(
                results.query(
                    "model == 'gradient_boosting' "
                    "and representation == 'sorted_coulomb'"
                )["mae_ev"].iloc[0]
            ),
        ),
        ("Neural network\n(geometry)", validation_mae),
    ]

    labels = [label for label, _ in rows]
    values = [value for _, value in rows]
    colours = [BASELINE_COLOUR] * (len(rows) - 1) + [MODEL_COLOUR]

    figure, ax = plt.subplots(figsize=(8.5, 4.6), dpi=150)
    bars = ax.bar(labels, values, color=colours, width=0.62, zorder=3)

    for bar, value in zip(bars, values, strict=True):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            value + 0.02,
            f"{value:.3f}",
            ha="center",
            va="bottom",
            fontsize=10,
            fontweight="bold" if value == values[-1] else "normal",
            color=MODEL_COLOUR if value == values[-1] else "#5a6672",
        )

    style_axes(ax)
    ax.set_ylabel("Validation MAE (eV)   —   lower is better")
    ax.set_ylim(0, max(values) * 1.18)
    ax.set_title(
        "Each step earns its complexity",
        fontsize=13,
        fontweight="bold",
        loc="left",
        color="#1a1a1a",
        pad=32,
    )
    ax.text(
        0,
        1.02,
        "LUMO energy prediction on QM9 · same seeded split throughout",
        transform=ax.transAxes,
        fontsize=9.5,
        color="#5a6672",
    )
    figure.tight_layout()

    path = FIGURE_DIR / "model_comparison.png"
    figure.savefig(path, bbox_inches="tight", facecolor="white")
    plt.close(figure)
    return path


def figure_predicted_vs_true(true: np.ndarray, predicted: np.ndarray) -> Path:
    """Scatter of predicted against true LUMO, with the shrinkage made visible.

    The fitted line is a plain least-squares fit of predicted on true. A slope
    below 1 is the model regressing toward the mean: it under-predicts the high
    end and over-predicts the low end. Drawing it against the identity line is
    the whole point of the figure.
    """
    slope, intercept = np.polyfit(true, predicted, 1)
    errors = np.abs(predicted - true)
    mae = float(errors.mean())
    residual = float(((predicted - true) ** 2).sum())
    total = float(((true - true.mean()) ** 2).sum())
    r2 = 1.0 - residual / total

    figure, ax = plt.subplots(figsize=(6.4, 6.0), dpi=150)

    # Alpha plus small markers: with 13,083 points the interesting structure is
    # the density, which opaque markers would hide entirely.
    ax.scatter(
        true,
        predicted,
        s=4,
        alpha=0.15,
        color=MODEL_COLOUR,
        linewidths=0,
        zorder=3,
        rasterized=True,
    )

    limits = [
        min(true.min(), predicted.min()) - 0.3,
        max(true.max(), predicted.max()) + 0.3,
    ]
    ax.plot(
        limits,
        limits,
        color="#c0392b",
        linewidth=1.4,
        zorder=4,
        label="perfect prediction",
    )

    line_x = np.array(limits)
    ax.plot(
        line_x,
        slope * line_x + intercept,
        color="#e8a33d",
        linewidth=1.6,
        linestyle="--",
        zorder=5,
        label=f"fitted, slope = {slope:.3f}",
    )

    style_axes(ax)
    ax.set_xlim(limits)
    ax.set_ylim(limits)
    ax.set_aspect("equal")
    ax.set_xlabel("True LUMO (eV)")
    ax.set_ylabel("Predicted LUMO (eV)")
    ax.set_title(
        "Accurate overall, shrunk at the extremes",
        fontsize=13,
        fontweight="bold",
        loc="left",
        color="#1a1a1a",
        pad=32,
    )
    ax.text(
        0,
        1.02,
        f"{len(true):,} validation molecules · MAE {mae:.3f} eV · R² {r2:.3f}",
        transform=ax.transAxes,
        fontsize=9.5,
        color="#5a6672",
    )
    ax.legend(loc="upper left", frameon=False, fontsize=9.5)
    figure.tight_layout()

    path = FIGURE_DIR / "predicted_vs_true.png"
    figure.savefig(path, bbox_inches="tight", facecolor="white")
    plt.close(figure)
    return path


def main() -> None:
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)

    if not SERVED_ARTIFACT.exists():
        raise FileNotFoundError(
            f"{SERVED_ARTIFACT} not found. Run notebooks/04_pytorch.ipynb, or see "
            "the README."
        )

    model, scaler, metadata = load_artifact(SERVED_ARTIFACT, device="cpu")
    print(
        f"artifact: {metadata.representation}, {metadata.n_features} features, "
        f"validation MAE {metadata.validation_mae_ev:.4f} eV"
    )

    # The same split the checkpoint was trained under -- read from its metadata,
    # not passed in. Scoring against a different split would silently report a
    # number for molecules the model had already seen.
    frame = load_qm9()
    _, validation, _ = split_dataset(frame, seed=metadata.split_seed)
    print(f"validation molecules: {len(validation):,}")

    features = load_features(validation, kind=metadata.representation, use_cache=False)
    predicted = predict(model, scaler, features, device="cpu")
    true = validation[metadata.target].to_numpy()

    for path in (
        figure_model_comparison(metadata.validation_mae_ev),
        figure_predicted_vs_true(true, predicted),
    ):
        here = Path.cwd()
        shown = path.relative_to(here) if path.is_relative_to(here) else path
        print(f"wrote {shown}")


if __name__ == "__main__":
    main()

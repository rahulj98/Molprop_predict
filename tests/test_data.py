"""Tests for QM9 loading, parsing, and splitting.

These run without the 82 MB download: the ``methane`` fixture in
``conftest.py`` is the verbatim contents of a real archive member, so the
parser is tested against the real format rather than against an idealised
version of it.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from molecular_property_predictor.data import (
    DEFAULT_PROCESSED_DIR,
    DEFAULT_RAW_DIR,
    HARTREE_TO_EV,
    PROJECT_ROOT,
    _parse_float,
    geometry,
    load_uncharacterized,
    parse_xyz,
    split_dataset,
)

# --- Locations --------------------------------------------------------------


def test_default_data_paths_are_anchored_to_the_repository():
    """Regression: a relative default resolves against the working directory.

    A notebook launched from ``notebooks/`` then builds its own private copy of
    the dataset rather than reusing the one at the repository root -- which is
    exactly what happened before these defaults became absolute.
    """
    assert PROJECT_ROOT.is_absolute()
    assert (PROJECT_ROOT / "pyproject.toml").exists()
    assert DEFAULT_RAW_DIR == PROJECT_ROOT / "data" / "raw"
    assert DEFAULT_PROCESSED_DIR == PROJECT_ROOT / "data" / "processed"


# --- Numeric parsing --------------------------------------------------------


@pytest.mark.parametrize(
    ("token", "expected"),
    [
        ("1.234", 1.234),
        ("-0.3877", -0.3877),
        ("0.", 0.0),
        # Mathematica's exponent notation, which plain float() rejects.
        ("1.234*^-6", 1.234e-6),
        ("-5.0*^3", -5000.0),
    ],
)
def test_parse_float_handles_mathematica_exponents(token: str, expected: float):
    assert _parse_float(token) == pytest.approx(expected)


def test_plain_float_would_fail_on_mathematica_notation():
    """Documents *why* _parse_float exists, so nobody 'simplifies' it away."""
    with pytest.raises(ValueError):
        float("1.234*^-6")


# --- Structure parsing ------------------------------------------------------


def test_parse_xyz_reads_molecule_identity(methane):
    assert methane["index"] == 1
    assert methane["n_atoms"] == 5
    assert methane["elements"] == ["C", "H", "H", "H", "H"]
    assert methane["atomic_numbers"] == [6, 1, 1, 1, 1]
    assert methane["smiles_gdb17"] == "C"
    assert methane["smiles_relaxed"] == "C"
    assert methane["inchi_gdb17"] == "InChI=1S/CH4/h1H4"


def test_parse_xyz_converts_hartree_properties_to_ev(methane):
    assert methane["homo"] == pytest.approx(-0.3877 * HARTREE_TO_EV)
    assert methane["lumo"] == pytest.approx(0.1171 * HARTREE_TO_EV)


def test_parse_xyz_leaves_non_hartree_properties_unconverted(methane):
    """Cv is cal/mol/K and alpha is a0^3 -- neither is an energy."""
    assert methane["Cv"] == pytest.approx(6.469)
    assert methane["alpha"] == pytest.approx(13.21)


def test_parse_xyz_gap_is_consistent_with_homo_and_lumo(methane):
    """Internal consistency: the published gap must equal lumo - homo."""
    assert methane["gap"] == pytest.approx(methane["lumo"] - methane["homo"])


def test_parse_xyz_rejects_wrong_property_count(methane_xyz):
    """A truncated header must fail loudly, not silently mislabel columns."""
    truncated = methane_xyz.replace("\t6.469\t\n", "\n", 1)

    with pytest.raises(ValueError):
        parse_xyz(truncated)


def test_geometry_reshapes_coordinates_to_atoms_by_three(methane):
    atomic_numbers, coordinates = geometry(pd.Series(methane))

    assert coordinates.shape == (5, 3)
    assert atomic_numbers.shape == (5,)
    np.testing.assert_allclose(coordinates[0], [-0.0126981359, 1.0858041578, 0.0080009958])


def test_geometry_gives_physically_sensible_bond_lengths(methane):
    """Every C-H bond in methane should be about 1.09 A."""
    _, coordinates = geometry(pd.Series(methane))

    bond_lengths = np.linalg.norm(coordinates[1:] - coordinates[0], axis=1)
    np.testing.assert_allclose(bond_lengths, 1.09, atol=0.02)


# --- Uncharacterized list ---------------------------------------------------


def test_load_uncharacterized_skips_prose_header(tmp_path):
    path = tmp_path / "uncharacterized.txt"
    path.write_text(
        "\n"
        "List of molecules among the 133885 GDB9 molecules for which ...\n"
        "=========================================================\n"
        "#   Index      GDB17 SMILES      SMILES for B3LYP XYZ    D_IJ\n"
        "=========================================================\n"
        "      58        NC(=N)C#N        NC(=N)C#N           6.036217\n"
        "      61        NC(=N)C=O        NC(=N)C=O           5.631463\n",
        encoding="utf-8",
    )

    assert load_uncharacterized(path) == {58, 61}


# --- Splitting --------------------------------------------------------------


@pytest.fixture
def frame() -> pd.DataFrame:
    return pd.DataFrame({"index": range(1000), "lumo": np.arange(1000, dtype=float)})


def test_split_returns_requested_proportions(frame):
    train, validation, test = split_dataset(frame, train_frac=0.8, val_frac=0.1)

    assert (len(train), len(validation), len(test)) == (800, 100, 100)


def test_split_partitions_every_row_exactly_once(frame):
    """No row may be dropped, and none may appear in two sets."""
    train, validation, test = split_dataset(frame)

    combined = pd.concat([train, validation, test])["index"]
    assert len(combined) == len(frame)
    assert set(combined) == set(frame["index"])


def test_split_is_disjoint(frame):
    train, validation, test = split_dataset(frame)

    assert not set(train["index"]) & set(validation["index"])
    assert not set(train["index"]) & set(test["index"])
    assert not set(validation["index"]) & set(test["index"])


def test_split_is_reproducible_for_a_given_seed(frame):
    """Phase 3 and Phase 4 results are only comparable if this holds."""
    first, _, _ = split_dataset(frame, seed=42)
    second, _, _ = split_dataset(frame, seed=42)

    pd.testing.assert_frame_equal(first, second)


def test_split_actually_shuffles(frame):
    """A different seed must give a different partition, or the seed is a lie."""
    first, _, _ = split_dataset(frame, seed=0)
    second, _, _ = split_dataset(frame, seed=1)

    assert list(first["index"]) != list(second["index"])


def test_split_does_not_merely_take_the_first_rows(frame):
    """Guards against an unshuffled split, which would correlate with index."""
    train, _, _ = split_dataset(frame, seed=0)

    assert list(train["index"]) != list(range(len(train)))


@pytest.mark.parametrize(
    ("train_frac", "val_frac"),
    [(0.95, 0.10), (1.0, 0.1), (0.8, 0.0), (-0.1, 0.5)],
)
def test_split_rejects_fractions_that_leave_no_test_set(frame, train_frac, val_frac):
    with pytest.raises(ValueError):
        split_dataset(frame, train_frac=train_frac, val_frac=val_frac)

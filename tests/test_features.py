"""Tests for the molecular representations.

The tests that matter here are the *invariance* tests. A representation that
silently loses rotation or permutation invariance still produces plausible
numbers and still trains -- it just trains worse, for a reason nobody would
find by reading the output. These assertions are the only thing standing
between us and that.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from molecular_property_predictor.data import geometry
from molecular_property_predictor.features import (
    N_MAX_ATOMS,
    composition_features,
    coulomb_matrix,
    eigenspectrum_features,
    featurize,
    load_features,
    pad_to,
    sort_by_row_norm,
    sorted_coulomb_features,
    upper_triangle,
)


@pytest.fixture
def methane_geometry(methane) -> tuple[np.ndarray, np.ndarray]:
    return geometry(pd.Series(methane))


def random_rotation(seed: int) -> np.ndarray:
    """A proper rotation matrix (orthogonal, determinant +1)."""
    generator = np.random.default_rng(seed)
    rotation, upper = np.linalg.qr(generator.normal(size=(3, 3)))
    rotation = rotation * np.sign(np.diag(upper))  # make the QR sign-unique
    if np.linalg.det(rotation) < 0:
        rotation[:, 0] *= -1
    return rotation


def test_random_rotation_is_orthogonal_with_unit_determinant():
    """The invariance tests are worthless if the helper is not a rotation."""
    rotation = random_rotation(0)

    np.testing.assert_allclose(rotation @ rotation.T, np.eye(3), atol=1e-12)
    assert np.linalg.det(rotation) == pytest.approx(1.0)


# --- The Coulomb matrix itself ----------------------------------------------


def test_coulomb_diagonal_is_half_z_to_the_2_4(methane_geometry):
    matrix = coulomb_matrix(*methane_geometry)

    assert matrix[0, 0] == pytest.approx(0.5 * 6**2.4)  # carbon
    np.testing.assert_allclose(np.diag(matrix)[1:], 0.5)  # four hydrogens


def test_coulomb_matrix_is_symmetric(methane_geometry):
    matrix = coulomb_matrix(*methane_geometry)

    np.testing.assert_allclose(matrix, matrix.T)


def test_coulomb_offdiagonal_is_nuclear_repulsion(methane_geometry):
    """Z_C * Z_H / r for a 1.09 A C-H bond, checked against the real distance."""
    atomic_numbers, coordinates = methane_geometry
    matrix = coulomb_matrix(atomic_numbers, coordinates)

    bond_length = np.linalg.norm(coordinates[1] - coordinates[0])
    assert matrix[0, 1] == pytest.approx(6 * 1 / bond_length)


def test_coulomb_matrix_is_invariant_under_rotation_and_translation(methane_geometry):
    """The whole reason this representation exists."""
    atomic_numbers, coordinates = methane_geometry
    moved = coordinates @ random_rotation(1).T + np.array([13.0, -2.5, 7.25])

    np.testing.assert_allclose(
        coulomb_matrix(atomic_numbers, moved),
        coulomb_matrix(atomic_numbers, coordinates),
        atol=1e-10,
    )


def test_coulomb_matrix_is_not_permutation_invariant(methane_geometry):
    """Documents *why* sort_by_row_norm exists, so nobody deletes it."""
    atomic_numbers, coordinates = methane_geometry
    order = [1, 0, 2, 3, 4]  # swap the carbon with the first hydrogen

    original = coulomb_matrix(atomic_numbers, coordinates)
    renumbered = coulomb_matrix(atomic_numbers[order], coordinates[order])

    assert not np.allclose(original, renumbered)


def test_coulomb_matrix_rejects_mismatched_lengths():
    with pytest.raises(ValueError):
        coulomb_matrix(np.array([6, 1]), np.zeros((3, 3)))


def test_coulomb_matrix_rejects_flat_coordinates():
    """Coordinates are cached flat; forgetting to reshape must fail loudly."""
    with pytest.raises(ValueError):
        coulomb_matrix(np.array([6, 1]), np.zeros(6))


# --- Permutation invariance by sorting --------------------------------------


def test_sort_by_row_norm_orders_rows_by_descending_norm(methane_geometry):
    sorted_matrix = sort_by_row_norm(coulomb_matrix(*methane_geometry))
    norms = np.linalg.norm(sorted_matrix, axis=1)

    assert np.all(np.diff(norms) <= 0)


def test_sorted_features_are_invariant_under_atom_permutation(methane_geometry):
    atomic_numbers, coordinates = methane_geometry
    order = [3, 1, 4, 0, 2]

    np.testing.assert_allclose(
        sorted_coulomb_features(atomic_numbers[order], coordinates[order]),
        sorted_coulomb_features(atomic_numbers, coordinates),
        atol=1e-10,
    )


def test_sorted_features_are_invariant_under_rotation(methane_geometry):
    atomic_numbers, coordinates = methane_geometry
    rotated = coordinates @ random_rotation(2).T

    np.testing.assert_allclose(
        sorted_coulomb_features(atomic_numbers, rotated),
        sorted_coulomb_features(atomic_numbers, coordinates),
        atol=1e-10,
    )


def test_sorted_features_have_the_expected_length(methane_geometry):
    """29 * 30 / 2 = 435."""
    assert sorted_coulomb_features(*methane_geometry).shape == (435,)


def test_padding_rows_sort_to_the_end(methane_geometry):
    """Methane fills 5 of 29 rows; the remaining 24 must be zero."""
    features = sorted_coulomb_features(*methane_geometry)

    matrix = np.zeros((N_MAX_ATOMS, N_MAX_ATOMS))
    row, column = np.triu_indices(N_MAX_ATOMS)
    matrix[row, column] = features

    assert np.count_nonzero(np.linalg.norm(matrix, axis=1)) == 5


# --- Padding and flattening -------------------------------------------------


def test_pad_to_places_the_molecule_in_the_top_left(methane_geometry):
    matrix = coulomb_matrix(*methane_geometry)
    padded = pad_to(matrix, N_MAX_ATOMS)

    assert padded.shape == (N_MAX_ATOMS, N_MAX_ATOMS)
    np.testing.assert_allclose(padded[:5, :5], matrix)
    assert np.count_nonzero(padded[5:, :]) == 0
    assert np.count_nonzero(padded[:, 5:]) == 0


def test_pad_to_rejects_a_molecule_larger_than_n_max():
    """Silently truncating a molecule would be far worse than failing."""
    with pytest.raises(ValueError):
        pad_to(np.ones((30, 30)), n_max=29)


def test_upper_triangle_round_trips_a_symmetric_matrix(methane_geometry):
    """Dropping the lower triangle must lose nothing."""
    matrix = coulomb_matrix(*methane_geometry)
    flat = upper_triangle(matrix)

    rebuilt = np.zeros_like(matrix)
    row, column = np.triu_indices(len(matrix))
    rebuilt[row, column] = flat
    rebuilt = rebuilt + rebuilt.T - np.diag(np.diag(rebuilt))

    assert flat.shape == (15,)  # 5 * 6 / 2
    np.testing.assert_allclose(rebuilt, matrix)


# --- Eigenspectrum ----------------------------------------------------------


def test_eigenspectrum_is_descending(methane_geometry):
    spectrum = eigenspectrum_features(*methane_geometry)

    assert spectrum.shape == (N_MAX_ATOMS,)
    assert np.all(np.diff(spectrum) <= 0)


def test_eigenspectrum_is_invariant_under_atom_permutation(methane_geometry):
    """Exactly invariant, not invariant-by-convention: eigenvalues do not care."""
    atomic_numbers, coordinates = methane_geometry
    order = [4, 2, 0, 3, 1]

    np.testing.assert_allclose(
        eigenspectrum_features(atomic_numbers[order], coordinates[order]),
        eigenspectrum_features(atomic_numbers, coordinates),
        atol=1e-10,
    )


def test_eigenspectrum_is_invariant_under_rotation(methane_geometry):
    atomic_numbers, coordinates = methane_geometry
    rotated = coordinates @ random_rotation(3).T

    np.testing.assert_allclose(
        eigenspectrum_features(atomic_numbers, rotated),
        eigenspectrum_features(atomic_numbers, coordinates),
        atol=1e-10,
    )


def test_padding_contributes_only_zero_eigenvalues(methane_geometry):
    """Padding must not distort the spectrum, only lengthen it.

    Note that most of the eigenvalues are *negative* -- one large positive one
    dominated by the carbon diagonal, and the rest below zero -- so the padding
    zeros land in the middle of the sorted spectrum, not at the end.
    """
    unpadded = np.linalg.eigvalsh(coulomb_matrix(*methane_geometry))
    padded = eigenspectrum_features(*methane_geometry)

    assert np.count_nonzero(np.abs(padded) > 1e-9) == 5
    expected = np.concatenate([unpadded, np.zeros(N_MAX_ATOMS - 5)])
    np.testing.assert_allclose(np.sort(padded), np.sort(expected), atol=1e-10)


# --- Composition control ----------------------------------------------------


def test_composition_counts_each_element(methane_geometry):
    """Methane is CH4: four H, one C, no N, O or F."""
    np.testing.assert_array_equal(
        composition_features(methane_geometry[0]), [4, 1, 0, 0, 0]
    )


def test_composition_ignores_geometry(methane_geometry):
    """The control must be blind to structure -- that is what it controls for."""
    atomic_numbers, coordinates = methane_geometry

    np.testing.assert_array_equal(
        composition_features(atomic_numbers, coordinates * 100.0),
        composition_features(atomic_numbers, coordinates),
    )


# --- Batch featurisation ----------------------------------------------------


@pytest.fixture
def frame(methane) -> pd.DataFrame:
    """Two rows: methane, and methane translated 5 A along x."""
    shifted = dict(methane)
    shifted["index"] = 2
    shifted["coordinates"] = [
        value + 5.0 if position % 3 == 0 else value
        for position, value in enumerate(methane["coordinates"])
    ]
    return pd.DataFrame([methane, shifted])


@pytest.mark.parametrize(
    ("kind", "n_features"),
    [("sorted_coulomb", 435), ("eigenspectrum", 29), ("composition", 5)],
)
def test_featurize_returns_one_row_per_molecule(frame, kind, n_features):
    features = featurize(frame, kind)

    assert features.shape == (2, n_features)
    assert features.dtype == np.float32


def test_featurize_is_blind_to_a_pure_translation(frame):
    """Two rows that differ only by a rigid shift must featurise identically."""
    features = featurize(frame, "sorted_coulomb")

    np.testing.assert_allclose(features[0], features[1], atol=1e-4)


def test_featurize_rejects_an_unknown_kind(frame):
    with pytest.raises(ValueError):
        featurize(frame, "morgan_fingerprint")


def test_load_features_writes_and_reuses_a_cache(frame, tmp_path):
    first = load_features(frame, "composition", tmp_path)
    assert (tmp_path / "features_composition.npz").exists()

    second = load_features(frame, "composition", tmp_path)
    np.testing.assert_array_equal(first, second)


def test_load_features_rebuilds_when_the_molecules_differ(frame, tmp_path):
    """A cache from one subset must never be reused for another."""
    load_features(frame, "composition", tmp_path)

    subset = frame.iloc[:1]
    assert load_features(subset, "composition", tmp_path).shape == (1, 5)

"""Turn molecules into fixed-length numeric vectors.

No model consumes a molecule. Every model from Phase 3 onward consumes a
fixed-length vector of floats, identical in length for a 3-atom molecule and a
29-atom one. This module is the map between the two, and it decides what the
model is capable of learning: whatever the encoding discards, no amount of
network depth recovers.

**The design constraint is invariance.** A molecule's properties do not change
if you translate it, rotate it, or renumber its atoms -- but the raw
``(n_atoms, 3)`` coordinate array changes under all three. Fed raw coordinates,
a model would have to learn from data that those three transformations are
meaningless, and would never entirely succeed. So the representation has to be
invariant *by construction*. That constraint is what produces the Coulomb
matrix below; the design is not arbitrary.

**Honest scope.** The Coulomb matrix (Rupp et al., *Phys. Rev. Lett.* **108**,
058301, 2012) is a 2012 representation, since superseded by SOAP, ACSF and
message-passing graph networks. Those reach far lower errors on QM9 than
anything here will. It is used because it is transparent, self-contained and
about forty lines of NumPy -- not because it is the best available.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from molecular_property_predictor.data import ATOMIC_NUMBERS, DEFAULT_PROCESSED_DIR

#: Largest molecule in QM9, measured from the parsed dataset rather than
#: assumed. Every feature vector is padded out to this size so that all
#: molecules yield the same length.
N_MAX_ATOMS = 29

#: Element order for :func:`composition_features`, by atomic number.
COMPOSITION_ELEMENTS: tuple[str, ...] = tuple(ATOMIC_NUMBERS)


def coulomb_matrix(
    atomic_numbers: np.ndarray, coordinates: np.ndarray
) -> np.ndarray:
    r"""Build the Coulomb matrix of one molecule.

    .. math::

        M_{ij} = \begin{cases}
            0.5\, Z_i^{2.4} & i = j \\
            Z_i Z_j / \lVert R_i - R_j \rVert & i \neq j
        \end{cases}

    The off-diagonal is the nuclear repulsion between two atoms, which depends
    only on the *distance* between them. That is where translation and rotation
    invariance come from -- for free, by construction. The diagonal is an
    empirical stand-in for the energy of the free atom; the 2.4 exponent was
    fitted, not derived, and nothing physical hangs on it.

    Note what this does *not* give us: renumbering the atoms permutes the rows
    and columns, so the matrix is not permutation invariant. See
    :func:`sort_by_row_norm`.

    Distances are in Angstrom, as parsed. The literature often works in Bohr;
    that choice only rescales every off-diagonal by one constant, which
    per-column standardisation in Phase 3 absorbs entirely. It does have one
    real effect: it changes the weight of the diagonal relative to the
    off-diagonals, and therefore the row ordering produced by
    :func:`sort_by_row_norm`.

    Args:
        atomic_numbers: Shape ``(n_atoms,)``.
        coordinates: Shape ``(n_atoms, 3)``, in Angstrom.

    Returns:
        Symmetric ``(n_atoms, n_atoms)`` array. Not padded, not sorted.
    """
    atomic_numbers = np.asarray(atomic_numbers, dtype=np.float64)
    coordinates = np.asarray(coordinates, dtype=np.float64)

    if coordinates.ndim != 2 or coordinates.shape[1] != 3:
        raise ValueError(f"coordinates must be (n_atoms, 3), got {coordinates.shape}")
    if len(atomic_numbers) != len(coordinates):
        raise ValueError(
            f"{len(atomic_numbers)} atomic numbers but "
            f"{len(coordinates)} coordinates"
        )

    displacements = coordinates[:, None, :] - coordinates[None, :, :]
    distances = np.linalg.norm(displacements, axis=-1)
    # The diagonal has its own formula, so setting the self-distance to
    # infinity zeroes it out here rather than dividing by zero.
    np.fill_diagonal(distances, np.inf)

    matrix = np.outer(atomic_numbers, atomic_numbers) / distances
    np.fill_diagonal(matrix, 0.5 * atomic_numbers**2.4)
    return matrix


def sort_by_row_norm(matrix: np.ndarray) -> np.ndarray:
    """Permute rows and columns by descending row norm.

    This is the standard fix for the Coulomb matrix's lack of permutation
    invariance: the row order is chosen by a property of the matrix itself, so
    the same molecule with its atoms renumbered sorts back to the same order.

    The cost is honest and worth knowing: the map is *discontinuous*. A small
    geometry change that swaps the norms of two rows swaps their positions in
    the vector, so two nearly identical molecules can sit far apart in feature
    space. This is the known weakness of the sorted Coulomb matrix, and part of
    why later representations replaced it.

    Args:
        matrix: Square, symmetric.

    Returns:
        The same matrix with a permutation applied to both axes.
    """
    order = np.argsort(-np.linalg.norm(matrix, axis=1))
    return matrix[np.ix_(order, order)]


def pad_to(matrix: np.ndarray, n_max: int = N_MAX_ATOMS) -> np.ndarray:
    """Zero-pad a square matrix out to ``(n_max, n_max)``.

    Padding is what makes the vectors a fixed length. The zero rows encode
    "no atom here"; the model has to learn to ignore them, which is a small
    price for uniform shape.

    Raises:
        ValueError: If the molecule is larger than ``n_max``. Better to fail
            loudly than to silently truncate a molecule.
    """
    n_atoms = len(matrix)
    if n_atoms > n_max:
        raise ValueError(f"molecule has {n_atoms} atoms, n_max is {n_max}")

    padded = np.zeros((n_max, n_max), dtype=matrix.dtype)
    padded[:n_atoms, :n_atoms] = matrix
    return padded


def upper_triangle(matrix: np.ndarray) -> np.ndarray:
    """Flatten the upper triangle, diagonal included.

    The matrix is symmetric, so the lower triangle is redundant. Dropping it
    takes 841 numbers down to 435 with no loss whatsoever.
    """
    row, column = np.triu_indices(len(matrix))
    return matrix[row, column]


# --- Per-molecule feature vectors -------------------------------------------
#
# All three share the signature ``(atomic_numbers, coordinates) -> 1-D array``
# so that `featurize` can dispatch between them.


def sorted_coulomb_features(
    atomic_numbers: np.ndarray,
    coordinates: np.ndarray,
    n_max: int = N_MAX_ATOMS,
) -> np.ndarray:
    """Flattened upper triangle of the padded, row-sorted Coulomb matrix.

    Padding happens before sorting, which is safe: every real row has a
    strictly positive norm (the diagonal alone is at least 0.5), so the zero
    padding rows always sort to the end.

    Returns:
        Shape ``(n_max * (n_max + 1) // 2,)`` -- 435 for QM9.
    """
    matrix = coulomb_matrix(atomic_numbers, coordinates)
    return upper_triangle(sort_by_row_norm(pad_to(matrix, n_max)))


def eigenspectrum_features(
    atomic_numbers: np.ndarray,
    coordinates: np.ndarray,
    n_max: int = N_MAX_ATOMS,
) -> np.ndarray:
    """Eigenvalues of the padded Coulomb matrix, in descending order.

    A different trade against the sorted matrix. The eigenvalues of a matrix
    are unchanged by any simultaneous row/column permutation, so this is
    permutation invariant *exactly* rather than by convention, and it is
    continuous in the geometry -- no discontinuity to worry about.

    What it costs: 29 numbers instead of 435. The spectrum does not determine
    the matrix, so genuinely different molecules can share one. Phase 3
    measures whether that loss matters.

    Returns:
        Shape ``(n_max,)``.
    """
    matrix = pad_to(coulomb_matrix(atomic_numbers, coordinates), n_max)
    # eigvalsh, not eig: the matrix is real symmetric, so the eigenvalues are
    # real and the symmetric solver is both faster and free of imaginary dust.
    return np.sort(np.linalg.eigvalsh(matrix))[::-1]


def composition_features(
    atomic_numbers: np.ndarray,
    coordinates: np.ndarray | None = None,
) -> np.ndarray:
    """Count of each element: ``(n_H, n_C, n_N, n_O, n_F)``.

    Five numbers and no chemistry at all. This is the control. Phase 1 found
    that atom counts alone already explain R^2 = 0.4555 of the variance in
    ``lumo``, so if a geometry-based representation cannot beat this, the
    geometry is buying us nothing and we need to find that out immediately
    rather than after training a network.

    ``coordinates`` is accepted and deliberately ignored, so that this shares a
    signature with the geometric representations -- ignoring the geometry is
    the entire point of the control.
    """
    del coordinates  # Unused by design; see the docstring.
    atomic_numbers = np.asarray(atomic_numbers)
    return np.array(
        [np.count_nonzero(atomic_numbers == ATOMIC_NUMBERS[element])
         for element in COMPOSITION_ELEMENTS],
        dtype=np.float64,
    )


#: The representations `featurize` can build, by name.
FEATURE_KINDS = {
    "sorted_coulomb": sorted_coulomb_features,
    "eigenspectrum": eigenspectrum_features,
    "composition": composition_features,
}


def featurize(
    frame: pd.DataFrame,
    kind: str = "sorted_coulomb",
    *,
    dtype: type = np.float32,
) -> np.ndarray:
    """Build the feature matrix for a whole DataFrame of molecules.

    Note what this does *not* do: it does not standardise the columns. Feature
    scales here differ by an order of magnitude (a carbon diagonal is ~37, a
    C-H off-diagonal ~5.5), and models will need them scaled -- but a scaler
    fitted over the whole dataset leaks test-set statistics into training. The
    scaler therefore belongs in Phase 3, fitted on the training split alone.

    Args:
        frame: Rows as returned by :func:`~molecular_property_predictor.data.load_qm9`.
        kind: A key of :data:`FEATURE_KINDS`.
        dtype: ``float32`` by default -- the precision PyTorch works in, and
            half the memory of ``float64`` over 130k molecules.

    Returns:
        Shape ``(len(frame), n_features)``.
    """
    if kind not in FEATURE_KINDS:
        raise ValueError(f"unknown kind {kind!r}; expected one of {sorted(FEATURE_KINDS)}")
    build = FEATURE_KINDS[kind]

    # Reading the two columns directly rather than via `data.geometry`, which
    # takes a Series and is the convenience form for a single molecule.
    vectors = []
    for numbers, flat_coordinates in zip(
        frame["atomic_numbers"], frame["coordinates"], strict=True
    ):
        vectors.append(
            build(
                np.asarray(numbers, dtype=np.int64),
                np.asarray(flat_coordinates, dtype=np.float64).reshape(-1, 3),
            )
        )

    return np.asarray(vectors, dtype=dtype)


def load_features(
    frame: pd.DataFrame,
    kind: str = "sorted_coulomb",
    processed_dir: Path | str = DEFAULT_PROCESSED_DIR,
    *,
    use_cache: bool = True,
) -> np.ndarray:
    """Featurize with an on-disk cache, mirroring ``load_qm9``.

    The cache stores the molecule indices alongside the features and rebuilds
    if they do not match the frame it is handed. Without that check a cache
    built from one subset would be silently reused for another, quietly
    mislabelling every row.

    The default directory is anchored to the repository, not to the working
    directory, so a notebook and a script agree on where the cache lives.
    """
    cache = Path(processed_dir) / f"features_{kind}.npz"
    molecule_index = frame["index"].to_numpy()

    if use_cache and cache.exists():
        stored = np.load(cache)
        if np.array_equal(stored["index"], molecule_index):
            return stored["features"]

    features = featurize(frame, kind)
    cache.parent.mkdir(parents=True, exist_ok=True)
    np.savez(cache, features=features, index=molecule_index)
    return features

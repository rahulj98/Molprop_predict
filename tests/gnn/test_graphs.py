"""Tests for graph construction, the radial basis, and batching."""

from __future__ import annotations

import numpy as np
import pytest
import torch

from molecular_gnn.graphs import (
    DEFAULT_CUTOFF,
    build_graph,
    collate,
    cosine_cutoff,
    gaussian_basis,
    radius_graph,
)


def line_of_atoms(n: int, spacing: float = 1.5) -> tuple[np.ndarray, np.ndarray]:
    """``n`` hydrogens evenly spaced along x -- distances are known by hand."""
    numbers = np.ones(n, dtype=np.int64)
    positions = np.stack([np.arange(n) * spacing, np.zeros(n), np.zeros(n)], axis=1)
    return numbers, positions


# --- Radius graph -------------------------------------------------------------


def test_every_close_pair_is_connected_in_both_directions():
    """Three atoms 1.5 Å apart, all within the 5 Å cutoff: 3 * 2 = 6 edges."""
    _, positions = line_of_atoms(3)

    edge_index, distances = radius_graph(positions)

    assert edge_index.shape == (2, 6)
    assert len(distances) == 6


def test_no_atom_is_connected_to_itself():
    """A self-loop would put a spurious spike at zero distance."""
    _, positions = line_of_atoms(4)

    edge_index, _ = radius_graph(positions)

    assert not (edge_index[0] == edge_index[1]).any()


def test_distances_are_correct():
    _, positions = line_of_atoms(2, spacing=1.5)

    _, distances = radius_graph(positions)

    assert distances == pytest.approx([1.5, 1.5])


def test_pairs_beyond_the_cutoff_are_excluded():
    """Two atoms 6 Å apart, cutoff 5 Å: no edges at all."""
    _, positions = line_of_atoms(2, spacing=6.0)

    edge_index, _ = radius_graph(positions, cutoff=DEFAULT_CUTOFF)

    assert edge_index.shape == (2, 0)


def test_the_graph_does_not_depend_on_atom_ordering():
    """Permutation invariance, which is the reason for using a graph at all.

    Relabelling the atoms must not change the multiset of edge distances.
    """
    _, positions = line_of_atoms(5)
    permuted = positions[[3, 0, 4, 1, 2]]

    _, original_distances = radius_graph(positions)
    _, permuted_distances = radius_graph(permuted)

    assert sorted(original_distances) == pytest.approx(sorted(permuted_distances))


def test_the_graph_does_not_depend_on_position_or_orientation():
    """Translating and rotating a molecule must leave its graph unchanged."""
    _, positions = line_of_atoms(4)
    angle = 0.7
    rotation = np.array(
        [
            [np.cos(angle), -np.sin(angle), 0.0],
            [np.sin(angle), np.cos(angle), 0.0],
            [0.0, 0.0, 1.0],
        ]
    )
    moved = positions @ rotation.T + np.array([10.0, -3.0, 2.5])

    _, original = radius_graph(positions)
    _, transformed = radius_graph(moved)

    assert sorted(original) == pytest.approx(sorted(transformed))


def test_positions_of_the_wrong_shape_are_rejected():
    with pytest.raises(ValueError, match="shape"):
        radius_graph(np.zeros((4, 2)))


# --- Radial basis -------------------------------------------------------------


def test_basis_has_one_column_per_gaussian():
    values = gaussian_basis(np.array([1.0, 2.0, 3.0]), n_basis=16)

    assert values.shape == (3, 16)


def test_basis_peaks_at_the_centre_nearest_the_distance():
    """The expansion says which shell a neighbour is in."""
    n_basis = 32
    centres = np.linspace(0.0, DEFAULT_CUTOFF, n_basis)
    distance = 2.0

    values = gaussian_basis(np.array([distance]), n_basis=n_basis)

    assert np.argmax(values[0]) == np.argmin(np.abs(centres - distance))


def test_basis_values_lie_between_zero_and_one():
    values = gaussian_basis(np.linspace(0.0, DEFAULT_CUTOFF, 50))

    assert values.min() >= 0.0
    assert values.max() <= 1.0


def test_cutoff_envelope_decays_to_zero_at_the_boundary():
    """Without this, an atom crossing the cutoff changes the output abruptly."""
    assert cosine_cutoff(np.array([DEFAULT_CUTOFF]))[0] == pytest.approx(0.0, abs=1e-12)
    assert cosine_cutoff(np.array([0.0]))[0] == pytest.approx(1.0)


def test_cutoff_envelope_decreases_with_distance():
    values = cosine_cutoff(np.linspace(0.0, DEFAULT_CUTOFF, 20))

    assert np.all(np.diff(values) <= 0)


# --- Building and batching ----------------------------------------------------


def test_build_graph_records_atoms_edges_and_target(methane_geometry):
    numbers, positions = methane_geometry

    graph = build_graph(numbers, positions, target=3.19)

    assert graph.n_atoms == 5
    assert graph.n_edges == 20  # every ordered pair, all within 5 Å
    assert graph.target == pytest.approx(3.19)


def test_build_graph_rejects_mismatched_atoms_and_positions():
    with pytest.raises(ValueError, match="atomic numbers"):
        build_graph([1, 6], np.zeros((3, 3)))


def test_collate_offsets_each_graphs_edges(methane_geometry):
    """The one line in collate that fails silently if it is wrong.

    Without the offset, the second molecule's edges would address the first
    molecule's atoms: no error, just molecules wired to each other.
    """
    numbers, positions = methane_geometry
    graph = build_graph(numbers, positions, target=1.0)

    batch = collate([graph, graph])

    assert batch.atomic_numbers.shape == (10,)
    # The second graph's edges must all point into atoms 5..9.
    second_half = batch.edge_index[:, graph.n_edges :]
    assert second_half.min() >= 5
    assert batch.edge_index.max() < 10


def test_collate_labels_every_atom_with_its_molecule(methane_geometry):
    """The batch vector is what lets pooling put molecules back together."""
    numbers, positions = methane_geometry
    graph = build_graph(numbers, positions, target=1.0)

    batch = collate([graph, graph, graph])

    assert batch.batch.tolist() == [0] * 5 + [1] * 5 + [2] * 5
    assert batch.n_graphs == 3


def test_collate_stacks_targets_in_order(methane_geometry):
    numbers, positions = methane_geometry
    graphs = [
        build_graph(numbers, positions, target=value) for value in (1.0, -2.0, 3.5)
    ]

    batch = collate(graphs)

    assert batch.targets.tolist() == pytest.approx([1.0, -2.0, 3.5])


def test_collate_produces_edge_features_for_every_edge(methane_geometry):
    numbers, positions = methane_geometry
    graph = build_graph(numbers, positions, target=1.0)

    batch = collate([graph, graph], n_basis=16)

    assert batch.edge_basis.shape == (2 * graph.n_edges, 16)
    assert batch.edge_weight.shape == (2 * graph.n_edges,)
    assert batch.edge_basis.dtype == torch.float32


def test_collate_handles_molecules_of_different_sizes():
    """The reason batching needs machinery at all."""
    small = build_graph(*line_of_atoms(3), target=0.0)
    large = build_graph(*line_of_atoms(7), target=1.0)

    batch = collate([small, large])

    assert batch.atomic_numbers.shape == (10,)
    assert batch.batch.tolist() == [0] * 3 + [1] * 7
    assert batch.edge_index.shape[1] == small.n_edges + large.n_edges


def test_collate_rejects_an_empty_batch():
    with pytest.raises(ValueError, match="empty"):
        collate([])


def test_collate_rejects_a_partially_labelled_batch(methane_geometry):
    """Averaging a loss over whichever graphs happen to have a target hides a bug."""
    numbers, positions = methane_geometry
    labelled = build_graph(numbers, positions, target=1.0)
    unlabelled = build_graph(numbers, positions)

    with pytest.raises(ValueError, match="all one or all the other"):
        collate([labelled, unlabelled])


def test_collate_without_targets_is_allowed(methane_geometry):
    """Inference has no targets, and that is not an error."""
    numbers, positions = methane_geometry
    graph = build_graph(numbers, positions)

    batch = collate([graph])

    assert batch.targets is None

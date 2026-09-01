"""Turn a molecule into a graph, and many graphs into one batch.

**Why a graph at all.** The first package's answer was to flatten a molecule
into a fixed-length vector -- 435 numbers, padded to 29 atoms, rows sorted to
recover permutation invariance. That works, and it costs two things: the sort
makes the encoding discontinuous, and the padding means most of the vector is
zeros for a molecule of ordinary size. A graph representation gives up the
fixed length and gets both back. Atoms are nodes, neighbours are edges, and the
model that consumes it is *architecturally* invariant to atom ordering -- there
is no ordering to be invariant to.

**Why a radius graph rather than bonds.** QM9 ships 3D coordinates, and bonds
are not in the file; they would have to be perceived from the SMILES, which
throws away the geometry the coordinates encode. Connecting every pair of atoms
within a cutoff instead keeps the geometry, matches the input contract the
existing API already has, and is what SchNet and its descendants do. A cutoff
of 5 Å comfortably spans covalent bonds (roughly 1--1.5 Å) and the first few
shells of non-bonded neighbours.

**Why distances are expanded into a basis instead of used raw.** A single
distance is one number, and a network asked to learn a function of it starts
from a poor position: near-identical inputs must produce different filters, and
gradients through a raw distance are ill-conditioned. Expanding it onto a bank
of Gaussians centred at increasing distances turns "how far apart" into a
smooth, roughly one-hot-ish vector that says *which shell* the neighbour is in.
This is the continuous-filter idea from SchNet, and it is the single most
important detail in making distance-based message passing train at all.

**Why batching needs its own machinery.** Graphs have different numbers of
nodes, so they cannot be stacked into a rectangular tensor the way images or
fixed-length vectors can. The standard solution -- used by PyTorch Geometric
and reimplemented here -- is to concatenate all the nodes into one long tensor,
shift each graph's edge indices by the number of nodes already placed, and keep
a ``batch`` vector saying which graph each node belongs to. The batch is then
literally one big disconnected graph, and pooling at the end uses the ``batch``
vector to sum each molecule's nodes back together. No padding, no masking, and
the per-molecule cost does not depend on the largest molecule present.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np
import torch

#: Neighbour cutoff in Ångström. Large enough to include several shells around
#: each atom, small enough that the number of edges stays linear in practice
#: for molecules this size.
DEFAULT_CUTOFF = 5.0

#: Number of Gaussians in the radial basis.
DEFAULT_N_BASIS = 32


@dataclass(frozen=True)
class MolecularGraph:
    """One molecule as a graph.

    Attributes:
        atomic_numbers: ``(n_atoms,)`` integer nuclear charges.
        positions: ``(n_atoms, 3)`` Cartesian coordinates in Ångström.
        edge_index: ``(2, n_edges)``; ``edge_index[0]`` receives from
            ``edge_index[1]``. Both directions of every pair are present,
            because a message from i to j is not the message from j to i.
        edge_distance: ``(n_edges,)`` interatomic distances in Ångström.
        target: the property being predicted, or ``None`` for inference.
    """

    atomic_numbers: np.ndarray
    positions: np.ndarray
    edge_index: np.ndarray
    edge_distance: np.ndarray
    target: float | None = None

    @property
    def n_atoms(self) -> int:
        return len(self.atomic_numbers)

    @property
    def n_edges(self) -> int:
        return self.edge_index.shape[1]


def radius_graph(
    positions: np.ndarray, cutoff: float = DEFAULT_CUTOFF
) -> tuple[np.ndarray, np.ndarray]:
    """Connect every pair of atoms closer than ``cutoff``.

    Self-loops are excluded: an atom's distance to itself is zero, which would
    put a spurious spike at the bottom of the radial basis and let a node send
    itself a message that the update step already handles explicitly.

    The all-pairs distance matrix is computed in full. For molecules of up to
    29 atoms that is at most 841 entries, so a neighbour list or a spatial tree
    would be slower than the thing it optimises. This would be the wrong
    approach for a protein.

    Returns:
        ``(edge_index, distances)`` of shapes ``(2, n_edges)`` and
        ``(n_edges,)``.
    """
    positions = np.asarray(positions, dtype=np.float64)
    if positions.ndim != 2 or positions.shape[1] != 3:
        raise ValueError(
            f"positions must have shape (n_atoms, 3), got {positions.shape}"
        )

    deltas = positions[:, None, :] - positions[None, :, :]
    distances = np.linalg.norm(deltas, axis=-1)

    within = (distances < cutoff) & ~np.eye(len(positions), dtype=bool)
    receivers, senders = np.nonzero(within)

    edge_index = np.stack([receivers, senders]).astype(np.int64)
    return edge_index, distances[receivers, senders]


def gaussian_basis(
    distances: np.ndarray,
    n_basis: int = DEFAULT_N_BASIS,
    cutoff: float = DEFAULT_CUTOFF,
    *,
    start: float = 0.0,
) -> np.ndarray:
    """Expand distances onto a bank of evenly spaced Gaussians.

    The centres are spread from ``start`` to ``cutoff``, and the width is set to
    the spacing between them. Wider would blur neighbouring shells together;
    narrower would leave dead zones between centres where the gradient of every
    basis function is effectively zero, which is a quiet way to make a model
    untrainable.

    Returns:
        ``(n_edges, n_basis)``.
    """
    centres = np.linspace(start, cutoff, n_basis)
    width = (cutoff - start) / max(n_basis - 1, 1)
    difference = np.asarray(distances, dtype=np.float64)[:, None] - centres[None, :]
    return np.exp(-((difference / width) ** 2))


def cosine_cutoff(distances: np.ndarray, cutoff: float = DEFAULT_CUTOFF) -> np.ndarray:
    """A smooth envelope that decays to exactly zero at the cutoff.

    Without it, an atom drifting across the cutoff boundary makes its
    contribution appear or vanish discontinuously -- the same class of defect as
    the row-sorting jump in the first package's representation, arrived at from
    a different direction. Multiplying every message by ``0.5 * (cos(pi d / r) +
    1)`` makes the transition continuous, so the model's output is a smooth
    function of geometry.
    """
    distances = np.asarray(distances, dtype=np.float64)
    return 0.5 * (np.cos(np.pi * np.clip(distances, 0.0, cutoff) / cutoff) + 1.0)


def build_graph(
    atomic_numbers: Sequence[int] | np.ndarray,
    positions: np.ndarray,
    *,
    cutoff: float = DEFAULT_CUTOFF,
    target: float | None = None,
) -> MolecularGraph:
    """Build the graph for one molecule."""
    numbers = np.asarray(atomic_numbers, dtype=np.int64)
    coordinates = np.asarray(positions, dtype=np.float64).reshape(-1, 3)

    if len(numbers) != len(coordinates):
        raise ValueError(
            f"{len(numbers)} atomic numbers but {len(coordinates)} positions"
        )

    edge_index, distances = radius_graph(coordinates, cutoff=cutoff)
    return MolecularGraph(
        atomic_numbers=numbers,
        positions=coordinates,
        edge_index=edge_index,
        edge_distance=distances,
        target=target,
    )


def graph_from_row(row, *, cutoff: float = DEFAULT_CUTOFF, target_column: str = "lumo"):
    """Build a graph from one row of the dataframe :mod:`molecular_gnn.data` loads."""
    numbers = np.asarray(row["atomic_numbers"], dtype=np.int64)
    positions = np.asarray(row["coordinates"], dtype=np.float64).reshape(-1, 3)
    target = float(row[target_column]) if target_column in row else None
    return build_graph(numbers, positions, cutoff=cutoff, target=target)


@dataclass(frozen=True)
class GraphBatch:
    """Several graphs concatenated into one.

    Attributes:
        atomic_numbers: ``(total_atoms,)``.
        positions: ``(total_atoms, 3)``.
        edge_index: ``(2, total_edges)``, already offset per graph.
        edge_basis: ``(total_edges, n_basis)`` radial basis features.
        edge_weight: ``(total_edges,)`` the smooth cutoff envelope.
        batch: ``(total_atoms,)`` mapping each atom to its molecule's position
            in the batch. This is what pooling uses to put molecules back
            together, and it is the whole trick.
        targets: ``(n_graphs,)`` or ``None``.
        n_graphs: how many molecules are in the batch.
    """

    atomic_numbers: torch.Tensor
    positions: torch.Tensor
    edge_index: torch.Tensor
    edge_basis: torch.Tensor
    edge_weight: torch.Tensor
    batch: torch.Tensor
    targets: torch.Tensor | None
    n_graphs: int

    def to(self, device: torch.device | str) -> GraphBatch:
        """Move every tensor to ``device``, returning a new batch."""
        return GraphBatch(
            atomic_numbers=self.atomic_numbers.to(device),
            positions=self.positions.to(device),
            edge_index=self.edge_index.to(device),
            edge_basis=self.edge_basis.to(device),
            edge_weight=self.edge_weight.to(device),
            batch=self.batch.to(device),
            targets=None if self.targets is None else self.targets.to(device),
            n_graphs=self.n_graphs,
        )


def collate(
    graphs: Sequence[MolecularGraph],
    *,
    n_basis: int = DEFAULT_N_BASIS,
    cutoff: float = DEFAULT_CUTOFF,
) -> GraphBatch:
    """Concatenate graphs into one batch.

    The only subtle line is the edge offset. Each graph's ``edge_index`` refers
    to its own atoms, numbered from zero; after concatenation those numbers
    address the wrong atoms unless each graph's indices are shifted by the
    number of atoms already placed. Getting this wrong does not raise -- it
    silently wires molecules to each other, and the model trains to a worse
    score for no visible reason.

    Raises:
        ValueError: If ``graphs`` is empty, or if only some carry a target.
            A batch with a partial target is a bug in the caller, and averaging
            a loss over the ones that happen to have one would hide it.
    """
    if not graphs:
        raise ValueError("cannot collate an empty sequence of graphs")

    with_target = [graph.target is not None for graph in graphs]
    if any(with_target) and not all(with_target):
        raise ValueError(
            "some graphs carry a target and some do not; a batch must be all "
            "one or all the other"
        )

    atomic_numbers = np.concatenate([graph.atomic_numbers for graph in graphs])
    positions = np.concatenate([graph.positions for graph in graphs])
    distances = np.concatenate([graph.edge_distance for graph in graphs])

    offsets = np.cumsum([0] + [graph.n_atoms for graph in graphs[:-1]])
    edge_index = np.concatenate(
        [
            graph.edge_index + offset
            for graph, offset in zip(graphs, offsets, strict=True)
        ],
        axis=1,
    )

    batch = np.concatenate(
        [np.full(graph.n_atoms, position, dtype=np.int64)
         for position, graph in enumerate(graphs)]
    )

    targets = (
        torch.tensor([graph.target for graph in graphs], dtype=torch.float32)
        if all(with_target)
        else None
    )

    return GraphBatch(
        atomic_numbers=torch.from_numpy(atomic_numbers),
        positions=torch.from_numpy(positions).float(),
        edge_index=torch.from_numpy(edge_index),
        edge_basis=torch.from_numpy(
            gaussian_basis(distances, n_basis=n_basis, cutoff=cutoff)
        ).float(),
        edge_weight=torch.from_numpy(cosine_cutoff(distances, cutoff=cutoff)).float(),
        batch=torch.from_numpy(batch),
        targets=targets,
        n_graphs=len(graphs),
    )

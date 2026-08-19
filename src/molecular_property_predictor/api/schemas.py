"""The request and response contract, declared once as typed classes.

**What Pydantic is doing here, and why it is not just type-hint decoration.**
A schema class is a single declaration from which three things are derived: the
runtime validator that rejects malformed requests, the OpenAPI document that
powers ``/docs``, and the types the endpoint functions are written against. The
obvious cheaper alternative -- reading a dict and writing ``if "coordinates"
not in body`` by hand -- gives you the first one only, and it drifts out of
agreement with the documentation the first time anything changes.

**Why validation matters more for a model than for an ordinary web service.**
A malformed request to a database-backed API usually fails loudly: the row is
missing, the query errors. A malformed request to a *model* does not. Feed this
network a 434-long vector and it raises; feed it a 435-long vector of nonsense
and it returns a confident number that means nothing. The failure is silent, so
the boundary has to be strict on purpose. That is the same argument
:func:`~molecular_property_predictor.model.predict` already makes for taking
raw features rather than scaled ones -- forgetting is silent, so the interface
should make forgetting impossible.

**The contract is geometry, not features.** A caller has a molecule; it does not
have the flattened upper triangle of a sorted Coulomb matrix. Accepting the
435-vector would be a smaller service and a worse interface: it would push
featurisation onto every client, and it would weld the current representation
into the public contract, so replacing the descriptor later would break every
caller. Featurising server-side keeps the representation an implementation
detail, which is what it should be.

**What this does not validate.** These checks establish that a request is
*well-formed and numerically safe to featurise*. They do not establish that the
geometry is chemically sensible, and they deliberately do not try. See
:data:`MIN_INTERATOMIC_DISTANCE`.
"""

from __future__ import annotations

import math

import numpy as np
from pydantic import BaseModel, Field, model_validator

from molecular_property_predictor.data import ATOMIC_NUMBERS
from molecular_property_predictor.features import N_MAX_ATOMS

#: The five elements QM9 contains, as atomic numbers: H, C, N, O, F.
#:
#: A request naming anything else is rejected rather than served. The network
#: has never seen another element, and the Coulomb matrix would happily build a
#: feature vector for one -- the diagonal ``0.5 * Z**2.4`` is defined for any
#: ``Z``. It would simply be far outside the range the scaler was fitted on, and
#: the prediction would be meaningless without being detectably wrong.
ALLOWED_ATOMIC_NUMBERS = frozenset(ATOMIC_NUMBERS.values())

#: Rejection floor for the distance between two *distinct* atoms, in Angstrom.
#:
#: This is a numerical guard, not a chemical one, and the distinction is worth
#: keeping honest. :func:`~molecular_property_predictor.features.coulomb_matrix`
#: fills the diagonal separately, so an atom's distance to *itself* is handled --
#: but two distinct atoms at the same coordinates give ``Z_i * Z_j / 0``, which
#: is ``inf``, which propagates through the sort and into the network as an
#: infinite feature. NumPy emits a warning at most; the response would be
#: ``NaN`` or nonsense.
#:
#: 0.1 A sits far below any real bond (the shortest in QM9 is H-H at roughly
#: 0.74 A), so this rejects degenerate geometry without pretending to judge
#: chemistry. A structure that passes this check can still be physically
#: absurd, and the service does not claim otherwise.
MIN_INTERATOMIC_DISTANCE = 0.1

#: Methane, taken verbatim from ``dsgdb9nsd_000001.xyz``. Used as the worked
#: example in the generated OpenAPI docs so that ``/docs`` is immediately
#: usable -- a real molecule from the training distribution rather than an
#: invented one that would not survive the validators below.
METHANE_EXAMPLE = {
    "atomic_numbers": [6, 1, 1, 1, 1],
    "coordinates": [
        [-0.0126981359, 1.0858041578, 0.0080009958],
        [0.002150416, -0.0060313176, 0.0019761204],
        [1.0117308433, 1.4637511618, 0.0002765748],
        [-0.540815069, 1.4475266138, -0.8766437152],
        [-0.5238136345, 1.4379326443, 0.9063972942],
    ],
}


class Molecule(BaseModel):
    """One molecule, as a geometry.

    Coordinates are in Angstrom, matching the published QM9 files and therefore
    matching what the featuriser was built against. Units are stated rather than
    assumed because getting them wrong is exactly the kind of error that
    produces a plausible number instead of an exception: Bohr instead of
    Angstrom would rescale every off-diagonal by 1.89 and quietly change the row
    ordering the sorted representation depends on.
    """

    model_config = {"json_schema_extra": {"examples": [METHANE_EXAMPLE]}}

    atomic_numbers: list[int] = Field(
        ...,
        min_length=1,
        max_length=N_MAX_ATOMS,
        description="Atomic number per atom. QM9 covers H, C, N, O, F only.",
    )
    # A fixed-length tuple rather than `list[float]`, which gets the "exactly
    # three components" check from the type itself instead of a hand-written
    # one, and documents the shape in the OpenAPI schema.
    coordinates: list[tuple[float, float, float]] = Field(
        ...,
        min_length=1,
        max_length=N_MAX_ATOMS,
        description="Cartesian coordinates per atom, in Angstrom.",
    )

    @model_validator(mode="after")
    def _check_geometry(self) -> Molecule:
        """Reject anything that would featurise into garbage.

        Runs after per-field parsing, because every check here is a statement
        about the two lists *together* rather than about either one alone.
        """
        if len(self.atomic_numbers) != len(self.coordinates):
            raise ValueError(
                f"{len(self.atomic_numbers)} atomic numbers but "
                f"{len(self.coordinates)} coordinates: one entry per atom is "
                "required"
            )

        unknown = sorted(set(self.atomic_numbers) - ALLOWED_ATOMIC_NUMBERS)
        if unknown:
            raise ValueError(
                f"unsupported atomic numbers {unknown}; this model was trained "
                f"on QM9, which contains only {sorted(ALLOWED_ATOMIC_NUMBERS)} "
                "(H, C, N, O, F)"
            )

        # JSON has no NaN or Infinity literals, but `1e400` parses to inf and
        # some clients emit bare NaN regardless. Either would pass straight
        # through the featuriser into the response.
        for position, point in enumerate(self.coordinates):
            if not all(math.isfinite(value) for value in point):
                raise ValueError(
                    f"coordinates for atom {position} are not finite: {list(point)}"
                )

        self._check_no_coincident_atoms()
        return self

    def _check_no_coincident_atoms(self) -> None:
        """Reject distinct atoms sitting on top of one another.

        See :data:`MIN_INTERATOMIC_DISTANCE` for why this is a division-by-zero
        guard rather than a plausibility check.
        """
        if len(self.coordinates) < 2:
            return

        points = np.asarray(self.coordinates, dtype=np.float64)
        distances = np.linalg.norm(points[:, None, :] - points[None, :, :], axis=-1)
        # Only distinct pairs: the diagonal is zero by construction and is
        # handled separately by the Coulomb matrix.
        np.fill_diagonal(distances, np.inf)

        closest = float(distances.min())
        if closest < MIN_INTERATOMIC_DISTANCE:
            i, j = np.unravel_index(distances.argmin(), distances.shape)
            raise ValueError(
                f"atoms {int(i)} and {int(j)} are {closest:.4g} A apart, below "
                f"the {MIN_INTERATOMIC_DISTANCE} A floor; the Coulomb matrix "
                "divides by this distance"
            )

    def to_arrays(self) -> tuple[np.ndarray, np.ndarray]:
        """Convert to the ``(atomic_numbers, coordinates)`` pair the featuriser takes.

        The translation from wire types to NumPy lives here, at the edge, so
        that :mod:`~molecular_property_predictor.api.service` never handles a
        Pydantic object and the featuriser never handles a request. Same shape
        as :func:`~molecular_property_predictor.data.geometry`, which does this
        for a DataFrame row.

        Returns:
            ``(n_atoms,)`` int64 and ``(n_atoms, 3)`` float64, in Angstrom.
        """
        return (
            np.asarray(self.atomic_numbers, dtype=np.int64),
            np.asarray(self.coordinates, dtype=np.float64),
        )


class BatchRequest(BaseModel):
    """Several molecules in one call.

    Batching exists because the per-request overhead -- HTTP, validation, moving
    a tensor to the device -- is a large fraction of the cost for a network this
    small. Screening a candidate library one HTTP request at a time is the
    use case this project's README describes, so the API should support it in
    one round trip rather than a thousand.
    """

    model_config = {
        "json_schema_extra": {"examples": [{"molecules": [METHANE_EXAMPLE]}]}
    }

    molecules: list[Molecule] = Field(
        ...,
        min_length=1,
        # An upper bound because an unbounded list is a way to exhaust the
        # memory of whatever free-tier instance Phase 8 lands on. 1,000 sorted
        # Coulomb vectors is roughly 1.7 MB as float32, which is comfortable.
        max_length=1_000,
        description="One entry per molecule; all are scored in a single pass.",
    )


class Prediction(BaseModel):
    """The predicted property for one molecule.

    ``units`` is returned on every response rather than left to the
    documentation. The number is meaningless without it, and a caller that
    assumes Hartree because the QM9 paper uses Hartree would be wrong by a
    factor of 27.2 -- while getting a perfectly plausible-looking value.
    """

    lumo_ev: float = Field(
        ..., description="Predicted energy of the lowest unoccupied molecular orbital."
    )
    units: str = Field(default="eV", description="Units of `lumo_ev`.")
    n_atoms: int = Field(..., description="Atoms in the submitted geometry.")


class BatchPrediction(BaseModel):
    """Predictions for a batch, in the order submitted."""

    predictions: list[Prediction]


class ModelInfo(BaseModel):
    """What is actually being served.

    This endpoint is the honesty surface of the whole project. It reports the
    checkpoint's real provenance -- which representation, which seeds, how it
    scored -- so that a number returned by ``/predict`` can be traced back to
    the run that produced it. A served model whose training conditions are
    undocumented cannot be audited by whoever is relying on it.
    """

    target: str
    units: str
    representation: str
    n_features: int
    n_parameters: int
    hidden_sizes: list[int]
    dropout: float
    loss_name: str
    split_seed: int
    torch_seed: int
    epochs_trained: int
    best_epoch: int
    validation_mae_ev: float
    test_mae_ev: float | None = Field(
        default=None,
        description=(
            "Mean absolute error on the held-out test split, which is opened "
            "exactly once for the checkpoint that ships. Null on any artifact "
            "predating that evaluation."
        ),
    )
    max_atoms: int = Field(
        default=N_MAX_ATOMS,
        description="Largest molecule the representation can encode.",
    )
    supported_elements: list[int] = Field(
        default_factory=lambda: sorted(ALLOWED_ATOMIC_NUMBERS),
        description="Atomic numbers this model accepts.",
    )


class Health(BaseModel):
    """Liveness, and whether a model is actually loaded.

    Two separate facts, deliberately. Phases 7 and 8 need something that answers
    instantly without touching the model, but "the process is up" and "the
    process can serve predictions" are different claims -- a container that
    started fine and failed to find its checkpoint would otherwise report
    healthy and fail every request.
    """

    status: str
    model_loaded: bool

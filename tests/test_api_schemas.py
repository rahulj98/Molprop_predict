"""Tests for the API's request/response contract.

These are the tests for the boundary, and the boundary is where this project is
most exposed. A network handed a well-formed but wrong feature vector does not
raise -- it returns a confident number. So every test below that asserts a
*rejection* is guarding against a silent wrong answer rather than against a
crash, which is why there are more of them than the schema's size suggests.

The rejections are grouped by what makes them necessary: shape, chemistry,
and arithmetic.
"""

from __future__ import annotations

import numpy as np
import pytest
from pydantic import ValidationError

from molecular_property_predictor.api.schemas import (
    ALLOWED_ATOMIC_NUMBERS,
    METHANE_EXAMPLE,
    MIN_INTERATOMIC_DISTANCE,
    BatchRequest,
    Health,
    ModelInfo,
    Molecule,
    Prediction,
)
from molecular_property_predictor.data import ATOMIC_NUMBERS
from molecular_property_predictor.features import (
    N_MAX_ATOMS,
    sorted_coulomb_features,
)


@pytest.fixture
def methane() -> Molecule:
    return Molecule(**METHANE_EXAMPLE)


def line_of_atoms(n: int, spacing: float = 1.5) -> dict:
    """``n`` hydrogens strung out along x, comfortably far apart."""
    return {
        "atomic_numbers": [1] * n,
        "coordinates": [[i * spacing, 0.0, 0.0] for i in range(n)],
    }


# --- Accepting valid input --------------------------------------------------


def test_methane_example_validates(methane):
    """The example advertised in `/docs` must itself be a legal request.

    A worked example that fails its own validators is worse than no example:
    it is the first thing anyone tries.
    """
    assert len(methane.atomic_numbers) == 5
    assert methane.atomic_numbers[0] == ATOMIC_NUMBERS["C"]


def test_to_arrays_returns_featuriser_shapes(methane):
    atomic_numbers, coordinates = methane.to_arrays()

    assert atomic_numbers.shape == (5,)
    assert coordinates.shape == (5, 3)
    assert atomic_numbers.dtype == np.int64
    assert coordinates.dtype == np.float64


def test_to_arrays_output_feeds_the_real_featuriser(methane):
    """The point of the whole schema: what comes out must featurise.

    Asserting the length is 435 rather than merely "it did not raise" pins the
    contract the network was trained against.
    """
    features = sorted_coulomb_features(*methane.to_arrays())

    assert features.shape == (N_MAX_ATOMS * (N_MAX_ATOMS + 1) // 2,)
    assert np.all(np.isfinite(features))


def test_single_atom_is_accepted():
    """One atom has no interatomic distances, so the pair check must not fire."""
    molecule = Molecule(atomic_numbers=[9], coordinates=[[0.0, 0.0, 0.0]])

    assert np.all(np.isfinite(sorted_coulomb_features(*molecule.to_arrays())))


def test_largest_representable_molecule_is_accepted():
    """Exactly `N_MAX_ATOMS` fits; the boundary is inclusive."""
    molecule = Molecule(**line_of_atoms(N_MAX_ATOMS))

    assert len(molecule.atomic_numbers) == N_MAX_ATOMS


@pytest.mark.parametrize("atomic_number", sorted(ALLOWED_ATOMIC_NUMBERS))
def test_every_qm9_element_is_accepted(atomic_number):
    molecule = Molecule(
        atomic_numbers=[atomic_number], coordinates=[[0.0, 0.0, 0.0]]
    )

    assert molecule.atomic_numbers == [atomic_number]


# --- Rejecting bad shape ----------------------------------------------------


def test_mismatched_lengths_are_rejected():
    """Two atomic numbers and one coordinate is not a molecule.

    `coulomb_matrix` raises on this too, but only after the request has been
    accepted -- which turns a client error into a server error.
    """
    with pytest.raises(ValidationError, match="2 atomic numbers but 1 coordinates"):
        Molecule(atomic_numbers=[6, 1], coordinates=[[0.0, 0.0, 0.0]])


def test_empty_molecule_is_rejected():
    with pytest.raises(ValidationError):
        Molecule(atomic_numbers=[], coordinates=[])


def test_two_component_coordinate_is_rejected():
    """A point needs three components; the tuple type enforces it."""
    with pytest.raises(ValidationError):
        Molecule(atomic_numbers=[6], coordinates=[[0.0, 0.0]])


def test_four_component_coordinate_is_rejected():
    with pytest.raises(ValidationError):
        Molecule(atomic_numbers=[6], coordinates=[[0.0, 0.0, 0.0, 0.0]])


def test_molecule_larger_than_the_representation_is_rejected():
    """30 atoms cannot be encoded, so this must fail rather than truncate.

    `pad_to` pads a matrix up to `N_MAX_ATOMS`; it does not crop one down. Left
    unvalidated this would raise deep inside the featuriser, on a request the
    service should never have accepted.
    """
    with pytest.raises(ValidationError):
        Molecule(**line_of_atoms(N_MAX_ATOMS + 1))


# --- Rejecting bad chemistry ------------------------------------------------


def test_element_outside_qm9_is_rejected():
    """Phosphorus is not in QM9, and the featuriser would not complain.

    `0.5 * Z**2.4` is defined for any Z, so a silicon or phosphorus diagonal
    would produce a feature vector far outside the scaler's fitted range and a
    number with no basis behind it.
    """
    with pytest.raises(ValidationError, match=r"unsupported atomic numbers \[15\]"):
        Molecule(
            atomic_numbers=[6, 15],
            coordinates=[[0.0, 0.0, 0.0], [1.5, 0.0, 0.0]],
        )


def test_rejection_message_names_every_unsupported_element():
    """Reporting one at a time makes fixing a request an iterative guess."""
    with pytest.raises(ValidationError, match=r"\[15, 16\]"):
        Molecule(
            atomic_numbers=[15, 16],
            coordinates=[[0.0, 0.0, 0.0], [1.5, 0.0, 0.0]],
        )


def test_zero_and_negative_atomic_numbers_are_rejected():
    with pytest.raises(ValidationError):
        Molecule(
            atomic_numbers=[0, -6],
            coordinates=[[0.0, 0.0, 0.0], [1.5, 0.0, 0.0]],
        )


# --- Rejecting bad arithmetic -----------------------------------------------


def test_coincident_atoms_are_rejected():
    """The case that would divide by zero and return `inf` or `NaN`."""
    with pytest.raises(ValidationError, match=r"below the .* floor"):
        Molecule(
            atomic_numbers=[6, 1],
            coordinates=[[0.0, 0.0, 0.0], [0.0, 0.0, 0.0]],
        )


def test_atoms_just_below_the_floor_are_rejected():
    with pytest.raises(ValidationError, match="apart"):
        Molecule(
            atomic_numbers=[6, 1],
            coordinates=[[0.0, 0.0, 0.0], [MIN_INTERATOMIC_DISTANCE * 0.9, 0.0, 0.0]],
        )


def test_atoms_above_the_floor_are_accepted():
    """The floor rejects degenerate geometry, not merely close geometry.

    Deliberately well below a real bond length: this check is a guard against
    division by zero, and it must not quietly become a chemical plausibility
    filter that rejects legitimate structures.
    """
    molecule = Molecule(
        atomic_numbers=[6, 1],
        coordinates=[[0.0, 0.0, 0.0], [MIN_INTERATOMIC_DISTANCE * 1.1, 0.0, 0.0]],
    )

    assert np.all(np.isfinite(sorted_coulomb_features(*molecule.to_arrays())))


def test_coincident_pair_is_found_among_many_valid_atoms():
    """The check is over all pairs, not just neighbours in the list."""
    payload = line_of_atoms(6)
    payload["coordinates"][5] = list(payload["coordinates"][0])

    with pytest.raises(ValidationError, match="atoms 0 and 5"):
        Molecule(**payload)


def test_infinite_coordinate_is_rejected():
    """`1e400` is legal JSON and parses to `inf`.

    Without this, the infinity flows through the Coulomb matrix into the
    response as a NaN, with a NumPy warning as the only signal.
    """
    with pytest.raises(ValidationError, match="not finite"):
        Molecule(
            atomic_numbers=[6, 1],
            coordinates=[[0.0, 0.0, 0.0], [1e400, 0.0, 0.0]],
        )


def test_nan_coordinate_is_rejected():
    with pytest.raises(ValidationError, match="not finite"):
        Molecule(
            atomic_numbers=[6, 1],
            coordinates=[[0.0, 0.0, 0.0], [float("nan"), 0.0, 0.0]],
        )


# --- The batch request ------------------------------------------------------


def test_batch_accepts_several_molecules():
    batch = BatchRequest(molecules=[METHANE_EXAMPLE, METHANE_EXAMPLE])

    assert len(batch.molecules) == 2
    assert isinstance(batch.molecules[0], Molecule)


def test_empty_batch_is_rejected():
    """An empty batch is a request that cannot be answered, not an empty answer."""
    with pytest.raises(ValidationError):
        BatchRequest(molecules=[])


def test_oversized_batch_is_rejected():
    with pytest.raises(ValidationError):
        BatchRequest(molecules=[METHANE_EXAMPLE] * 1_001)


def test_one_invalid_molecule_rejects_the_whole_batch():
    """Fail the request rather than silently scoring a subset.

    A partial response would need the caller to reconcile which inputs produced
    which outputs -- easy to get wrong, and wrong in the direction of
    misattributing a prediction to the wrong molecule.
    """
    with pytest.raises(ValidationError):
        BatchRequest(
            molecules=[METHANE_EXAMPLE, {"atomic_numbers": [6], "coordinates": []}]
        )


# --- Responses --------------------------------------------------------------


def test_prediction_defaults_to_ev():
    """Units are part of the payload, not part of the documentation."""
    prediction = Prediction(lumo_ev=-0.25, n_atoms=5)

    assert prediction.units == "eV"


def test_model_info_defaults_describe_the_accepted_input():
    """`/model` should tell a caller what `/predict` will accept."""
    info = ModelInfo(
        target="lumo",
        units="eV",
        representation="sorted_coulomb",
        n_features=435,
        n_parameters=387_585,
        hidden_sizes=[512, 256, 128],
        dropout=0.1,
        loss_name="mse",
        split_seed=0,
        torch_seed=0,
        epochs_trained=84,
        best_epoch=74,
        validation_mae_ev=0.2542,
    )

    assert info.max_atoms == N_MAX_ATOMS
    assert info.supported_elements == sorted(ALLOWED_ATOMIC_NUMBERS)
    # Null rather than absent: artifacts predating the test evaluation are
    # still loadable, and the field says "not measured" rather than "zero".
    assert info.test_mae_ev is None


def test_health_reports_liveness_and_readiness_separately():
    """A process that is up but has no model must not report itself ready."""
    health = Health(status="ok", model_loaded=False)

    assert health.status == "ok"
    assert health.model_loaded is False

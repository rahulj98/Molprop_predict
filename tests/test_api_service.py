"""Tests for the prediction service, with no HTTP anywhere in them.

That absence is the point. If these tests needed a client or a port, the split
between :mod:`~molecular_property_predictor.api.service` and
:mod:`~molecular_property_predictor.api.main` would not be real.

The artifact fixtures are built here rather than read from ``models/``, which is
gitignored: a test that depends on a build output is a test that fails on a
fresh clone.
"""

from __future__ import annotations

from dataclasses import FrozenInstanceError

import numpy as np
import pytest
import torch
from sklearn.preprocessing import StandardScaler

from molecular_property_predictor.api.service import (
    DEFAULT_ARTIFACT,
    MODEL_PATH_ENV,
    PredictionService,
    resolve_model_path,
)
from molecular_property_predictor.features import (
    N_MAX_ATOMS,
    sorted_coulomb_features,
)
from molecular_property_predictor.model import (
    ArtifactMetadata,
    MolecularNet,
    predict,
    save_artifact,
)

N_SORTED_COULOMB = N_MAX_ATOMS * (N_MAX_ATOMS + 1) // 2  # 435


def build_artifact(
    tmp_path,
    *,
    representation: str = "sorted_coulomb",
    n_features: int = N_SORTED_COULOMB,
    declared_n_features: int | None = None,
    name: str = "served.pt",
):
    """Write a small but structurally real checkpoint.

    Args:
        declared_n_features: What the *metadata* claims, when that needs to
            disagree with reality to test the consistency check.
    """
    torch.manual_seed(0)
    model = MolecularNet(n_features=n_features, hidden_sizes=(8, 4), dropout=0.1)

    generator = np.random.default_rng(0)
    scaler = StandardScaler().fit(generator.normal(size=(32, n_features)))

    metadata = ArtifactMetadata(
        representation=representation,
        target="lumo",
        n_features=declared_n_features or n_features,
        hidden_sizes=(8, 4),
        dropout=0.1,
        loss_name="mse",
        split_seed=0,
        torch_seed=0,
        epochs_trained=3,
        best_epoch=2,
        validation_mae_ev=0.5,
    )
    return save_artifact(tmp_path / name, model, scaler, metadata)


@pytest.fixture
def artifact(tmp_path):
    return build_artifact(tmp_path)


@pytest.fixture
def service(artifact) -> PredictionService:
    return PredictionService.from_artifact(artifact)


@pytest.fixture
def methane_geometry(methane) -> tuple[np.ndarray, np.ndarray]:
    """Methane's `(atomic_numbers, coordinates)`, from the real .xyz fixture."""
    return (
        np.asarray(methane["atomic_numbers"], dtype=np.int64),
        np.asarray(methane["coordinates"], dtype=np.float64).reshape(-1, 3),
    )


# --- Choosing which checkpoint to serve --------------------------------------


def test_explicit_path_wins(tmp_path, monkeypatch):
    """An argument beats the environment, so tests never inherit a developer's."""
    monkeypatch.setenv(MODEL_PATH_ENV, str(tmp_path / "from_env.pt"))

    assert resolve_model_path(tmp_path / "explicit.pt") == tmp_path / "explicit.pt"


def test_environment_variable_is_used_when_no_argument(tmp_path, monkeypatch):
    monkeypatch.setenv(MODEL_PATH_ENV, str(tmp_path / "from_env.pt"))

    assert resolve_model_path() == tmp_path / "from_env.pt"


def test_default_applies_when_environment_is_unset(monkeypatch):
    monkeypatch.delenv(MODEL_PATH_ENV, raising=False)

    assert resolve_model_path() == DEFAULT_ARTIFACT


def test_empty_environment_variable_falls_back_to_default(monkeypatch):
    """`MODEL_PATH=` in a .env file should not resolve to the current directory."""
    monkeypatch.setenv(MODEL_PATH_ENV, "")

    assert resolve_model_path() == DEFAULT_ARTIFACT


# --- Failing at start-up rather than per request ------------------------------


def test_missing_artifact_names_the_file_and_the_remedy(tmp_path):
    """The error a fresh clone will hit, so it has to be actionable."""
    missing = tmp_path / "absent.pt"

    with pytest.raises(FileNotFoundError, match=r"absent\.pt") as caught:
        PredictionService.from_artifact(missing)

    assert MODEL_PATH_ENV in str(caught.value)


def test_unknown_representation_is_refused(tmp_path):
    artifact = build_artifact(tmp_path, representation="bag_of_bonds")

    with pytest.raises(ValueError, match="bag_of_bonds"):
        PredictionService.from_artifact(artifact)


def test_metadata_disagreeing_with_the_featuriser_is_refused(tmp_path):
    """The check that stops an artifact serving plausible nonsense.

    Here the network takes 29 inputs and the metadata claims `sorted_coulomb`,
    which produces 435. Caught at load, not on the first request.
    """
    artifact = build_artifact(
        tmp_path, n_features=N_MAX_ATOMS, declared_n_features=N_MAX_ATOMS
    )

    with pytest.raises(ValueError, match="does not match"):
        PredictionService.from_artifact(artifact)


def test_eigenspectrum_artifact_loads_with_its_own_featuriser(tmp_path):
    """The representation comes from the artifact, so this must work unchanged."""
    artifact = build_artifact(
        tmp_path, representation="eigenspectrum", n_features=N_MAX_ATOMS
    )

    service = PredictionService.from_artifact(artifact)

    assert service.metadata.representation == "eigenspectrum"


# --- Predicting ---------------------------------------------------------------


def test_featurize_produces_one_row_per_molecule(service, methane_geometry):
    features = service.featurize([methane_geometry, methane_geometry])

    assert features.shape == (2, N_SORTED_COULOMB)


def test_predict_returns_one_value_per_molecule(service, methane_geometry):
    predictions = service.predict([methane_geometry] * 3)

    assert predictions.shape == (3,)
    assert np.all(np.isfinite(predictions))


def test_predict_one_returns_a_plain_float(service, methane_geometry):
    """A NumPy scalar would not serialise to JSON without coercion."""
    value = service.predict_one(methane_geometry)

    assert type(value) is float


def test_empty_batch_returns_empty_rather_than_raising(service):
    assert service.predict([]).shape == (0,)


def test_batching_does_not_change_any_prediction(service, methane_geometry):
    """A molecule must score the same alone as in company.

    If it did not, a caller's result would depend on what else they happened to
    submit alongside it -- and nothing would raise.
    """
    fluorine = (np.array([9, 9]), np.array([[0.0, 0.0, 0.0], [1.4, 0.0, 0.0]]))

    together = service.predict([methane_geometry, fluorine])
    apart = [service.predict_one(methane_geometry), service.predict_one(fluorine)]

    np.testing.assert_allclose(together, apart, rtol=0, atol=0)


def test_order_is_preserved(service, methane_geometry):
    """Predictions come back aligned to the submitted order.

    Misalignment here attributes a prediction to the wrong molecule, which is
    wrong in the worst possible way: quietly, and with plausible numbers.
    """
    hydrogen = (np.array([1]), np.array([[0.0, 0.0, 0.0]]))

    forward = service.predict([methane_geometry, hydrogen])
    backward = service.predict([hydrogen, methane_geometry])

    np.testing.assert_allclose(forward, backward[::-1], rtol=0, atol=0)


def test_service_matches_calling_predict_directly(service, methane_geometry):
    """The service must not perturb the numbers the model layer produces.

    This is the test that makes the whole API trustworthy: everything above is
    plumbing, and plumbing that changes the answer is worse than no API.
    """
    expected = predict(
        service.model,
        service.scaler,
        sorted_coulomb_features(*methane_geometry).reshape(1, -1),
        device="cpu",
    )

    np.testing.assert_allclose(
        service.predict([methane_geometry]), expected, rtol=0, atol=0
    )


def test_predictions_are_deterministic(service, methane_geometry):
    """Dropout must be off. `load_artifact` calls eval(), and this asserts it.

    Without eval() the service would return a different number for the same
    molecule on every call -- reproducibly wrong in a way a casual test that
    checks only "is it a float" would miss.
    """
    first = service.predict_one(methane_geometry)
    second = service.predict_one(methane_geometry)

    assert first == second


def test_service_is_frozen(service):
    """Nothing should swap the model out from under a concurrent request."""
    # Named rather than `Exception`: a bare `raises(Exception)` also passes if
    # the attribute assignment fails for some unrelated reason, which would let
    # this test keep passing after the freeze had been lost.
    with pytest.raises(FrozenInstanceError):
        service.model = None

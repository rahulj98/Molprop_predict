"""Tests for the HTTP endpoints, driven through FastAPI's TestClient.

``TestClient`` speaks to the application in-process -- no port is opened and no
server runs. That matters beyond convenience: it means these tests exercise the
real routing, validation and serialisation without any of it depending on the
network, so they behave identically here and in Phase 7's container build.

Using the client as a context manager is what triggers the ``lifespan`` handler.
Without the ``with`` block the model is never loaded and every test would run
against an unready application -- a quiet trap worth knowing about.
"""

from __future__ import annotations

import numpy as np
import pytest
from fastapi.testclient import TestClient

from molecular_property_predictor.api.main import create_app
from molecular_property_predictor.api.schemas import METHANE_EXAMPLE
from molecular_property_predictor.api.service import PredictionService
from molecular_property_predictor.features import N_MAX_ATOMS

from tests.test_api_service import build_artifact


@pytest.fixture
def client(tmp_path) -> TestClient:
    """An application serving a small artifact written to `tmp_path`."""
    artifact = build_artifact(tmp_path)
    with TestClient(create_app(artifact)) as test_client:
        yield test_client


@pytest.fixture
def modelless_client(tmp_path) -> TestClient:
    """An application whose checkpoint does not exist."""
    with TestClient(create_app(tmp_path / "absent.pt")) as test_client:
        yield test_client


def line_of_atoms(n: int, spacing: float = 1.5) -> dict:
    return {
        "atomic_numbers": [1] * n,
        "coordinates": [[i * spacing, 0.0, 0.0] for i in range(n)],
    }


# --- Meta endpoints -----------------------------------------------------------


def test_root_points_at_the_documentation(client):
    """A deployed URL opened in a browser should not return 404."""
    response = client.get("/")

    assert response.status_code == 200
    assert response.json()["docs"] == "/docs"


def test_health_reports_ready_when_a_model_is_loaded(client):
    response = client.get("/health")

    assert response.status_code == 200
    assert response.json() == {"status": "ok", "model_loaded": True}


def test_health_answers_even_with_no_model(modelless_client):
    """The case /health exists for: up, but unable to serve.

    A 200 with `model_loaded: false` rather than an error, because liveness and
    readiness are different claims and this endpoint reports both.
    """
    response = modelless_client.get("/health")

    assert response.status_code == 200
    assert response.json()["model_loaded"] is False


def test_model_endpoint_reports_the_artifact_provenance(client):
    response = client.get("/model")
    body = response.json()

    assert response.status_code == 200
    assert body["target"] == "lumo"
    assert body["units"] == "eV"
    assert body["representation"] == "sorted_coulomb"
    assert body["max_atoms"] == N_MAX_ATOMS
    assert body["supported_elements"] == [1, 6, 7, 8, 9]


def test_model_endpoint_reports_test_mae_as_null_when_unmeasured(client):
    """Null means "not measured", which is not the same as zero."""
    assert client.get("/model").json()["test_mae_ev"] is None


def test_openapi_schema_is_generated(client):
    """The docs come from the schemas, so they cannot drift out of date."""
    schema = client.get("/openapi.json").json()

    assert "/predict" in schema["paths"]
    assert "/predict/batch" in schema["paths"]


# --- Predicting ---------------------------------------------------------------


def test_predict_returns_a_number_with_its_units(client):
    response = client.post("/predict", json=METHANE_EXAMPLE)
    body = response.json()

    assert response.status_code == 200
    assert isinstance(body["lumo_ev"], float)
    assert body["units"] == "eV"
    assert body["n_atoms"] == 5


def test_predict_matches_the_service_directly(client, tmp_path):
    """The HTTP layer must not perturb the number the service computed.

    Everything between the request and the model is plumbing; plumbing that
    changes the answer is worse than having no API at all.
    """
    service = PredictionService.from_artifact(build_artifact(tmp_path))
    expected = service.predict_one(
        (
            np.asarray(METHANE_EXAMPLE["atomic_numbers"], dtype=np.int64),
            np.asarray(METHANE_EXAMPLE["coordinates"], dtype=np.float64),
        )
    )

    served = client.post("/predict", json=METHANE_EXAMPLE).json()["lumo_ev"]

    assert served == pytest.approx(expected, rel=0, abs=1e-12)


def test_batch_returns_one_prediction_per_molecule_in_order(client):
    hydrogen = {"atomic_numbers": [1], "coordinates": [[0.0, 0.0, 0.0]]}
    response = client.post(
        "/predict/batch", json={"molecules": [METHANE_EXAMPLE, hydrogen]}
    )
    predictions = response.json()["predictions"]

    assert response.status_code == 200
    assert [p["n_atoms"] for p in predictions] == [5, 1]


def test_batch_of_one_agrees_with_the_single_endpoint(client):
    """Two routes into the same computation must not disagree."""
    single = client.post("/predict", json=METHANE_EXAMPLE).json()["lumo_ev"]
    batched = client.post(
        "/predict/batch", json={"molecules": [METHANE_EXAMPLE]}
    ).json()["predictions"][0]["lumo_ev"]

    assert single == batched


def test_largest_representable_molecule_is_served(client):
    response = client.post("/predict", json=line_of_atoms(N_MAX_ATOMS))

    assert response.status_code == 200


# --- Rejecting bad requests ---------------------------------------------------


@pytest.mark.parametrize(
    "payload, reason",
    [
        ({"atomic_numbers": [6, 1], "coordinates": [[0.0, 0.0, 0.0]]}, "length mismatch"),
        ({"atomic_numbers": [], "coordinates": []}, "empty molecule"),
        ({"atomic_numbers": [6], "coordinates": [[0.0, 0.0]]}, "2D coordinate"),
        (
            {"atomic_numbers": [6, 15], "coordinates": [[0.0, 0.0, 0.0], [1.5, 0.0, 0.0]]},
            "element outside QM9",
        ),
        (
            {"atomic_numbers": [6, 1], "coordinates": [[0.0, 0.0, 0.0], [0.0, 0.0, 0.0]]},
            "coincident atoms",
        ),
        ({"atomic_numbers": [6]}, "missing coordinates"),
        ({}, "empty body"),
    ],
)
def test_malformed_requests_are_rejected_with_422(client, payload, reason):
    """422, not 500: these are client errors, and the body says which.

    Each of these would otherwise reach the featuriser and, in the worst cases,
    return a confident number rather than an error.
    """
    response = client.post("/predict", json=payload)

    assert response.status_code == 422, reason
    assert response.json()["detail"], reason


def test_oversized_molecule_is_rejected(client):
    """30 atoms cannot be encoded by a representation padded to 29."""
    response = client.post("/predict", json=line_of_atoms(N_MAX_ATOMS + 1))

    assert response.status_code == 422


def test_one_bad_molecule_rejects_the_whole_batch(client):
    """Better than a partial response the caller has to reconcile by index."""
    response = client.post(
        "/predict/batch",
        json={"molecules": [METHANE_EXAMPLE, {"atomic_numbers": [6], "coordinates": []}]},
    )

    assert response.status_code == 422


def test_empty_batch_is_rejected(client):
    response = client.post("/predict/batch", json={"molecules": []})

    assert response.status_code == 422


def test_wrong_method_is_rejected(client):
    """GET /predict is not a prediction; POST carries the body."""
    assert client.get("/predict").status_code == 405


# --- Behaviour with no model --------------------------------------------------


def test_predict_without_a_model_returns_503(modelless_client):
    """503, not 500: the request was fine, the service cannot answer it."""
    response = modelless_client.post("/predict", json=METHANE_EXAMPLE)

    assert response.status_code == 503


def test_the_503_explains_what_is_missing(modelless_client):
    """An operator should be able to diagnose this from the response alone."""
    detail = modelless_client.post("/predict", json=METHANE_EXAMPLE).json()["detail"]

    assert "MODEL_PATH" in detail


def test_model_endpoint_without_a_model_returns_503(modelless_client):
    assert modelless_client.get("/model").status_code == 503

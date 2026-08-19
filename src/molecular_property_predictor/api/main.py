"""The FastAPI application: endpoints, and the wiring that holds a model.

**What REST means here, concretely.** Resources are addressed by URL and acted
on with an HTTP verb. ``GET /model`` reads a description of what is loaded;
``POST /predict`` submits a molecule and gets a number back. ``GET`` because
reading is safe to repeat and cacheable; ``POST`` because a prediction request
carries a body and should not be cached against its URL. Errors travel as status
codes rather than as a ``200`` containing the word "error", which is what lets a
client distinguish "your request was wrong" (4xx) from "the service is broken"
(5xx) without parsing prose.

**Why the model is loaded in a lifespan handler.** ``lifespan`` runs once as the
process starts and once as it stops, so the checkpoint is read a single time and
held for the life of the application. See
:mod:`~molecular_property_predictor.api.service` for why doing it per request
would make loading the dominant cost of every call.

**Why a failed load does not crash the process.** The service validates the
artifact eagerly and raises immediately if it cannot serve -- but this module
catches that and starts anyway, in a state that reports itself unready.
``/health`` then answers honestly (``model_loaded: false``) and the prediction
endpoints return ``503`` naming the reason. The alternative, exiting on start-up,
is defensible and common; it was not chosen because a container that dies
instantly gives an operator nothing to query, whereas this one can be asked what
went wrong. The distinction being drawn is the standard one between *liveness*
(the process is running) and *readiness* (it can do useful work), which Phase 8's
host will care about.

**Why the endpoints are ``def`` and not ``async def``.** This is a real FastAPI
trap. A coroutine endpoint runs directly on the event loop, so blocking work
inside it stalls every other request in the process. Torch inference is exactly
that kind of blocking, CPU-bound work. Declared as a plain ``def``, FastAPI runs
the endpoint in a threadpool instead and the loop stays free. Writing
``async def`` here would look more modern and would serialise the whole service
under concurrent load.
"""

from __future__ import annotations

import logging
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import Depends, FastAPI, HTTPException, Request, status

from molecular_property_predictor.api.schemas import (
    BatchPrediction,
    BatchRequest,
    Health,
    ModelInfo,
    Molecule,
    Prediction,
)
from molecular_property_predictor.api.service import PredictionService

logger = logging.getLogger(__name__)

DESCRIPTION = """
Predicts the LUMO energy of a small organic molecule from its 3D geometry,
using a feed-forward network trained on the QM9 dataset.

**Scope, stated up front.** QM9's reference values were computed at
B3LYP/6-31G(2df,p) *relaxed* geometries, and this model was trained on those.
It therefore replaces the quantum-chemistry property calculation, not the
geometry optimisation that precedes it: submitting a rough or force-field
geometry will degrade accuracy silently. Predictions are only meaningful for
molecules resembling QM9 -- up to 29 atoms, built from H, C, N, O and F.

See `GET /model` for the served checkpoint's provenance and measured error.
"""


def create_app(model_path: str | Path | None = None) -> FastAPI:
    """Build the application.

    A factory rather than a bare module-level ``app`` so that tests can point an
    instance at a checkpoint in ``tmp_path``. Configuration passed as an
    argument beats configuration read from the environment inside the thing
    being tested: otherwise every test depends on the developer's shell.

    Args:
        model_path: Checkpoint to serve. Defaults to the ``MODEL_PATH``
            environment variable, then to the built-in default.
    """

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        """Load once on start-up; release on shutdown."""
        try:
            app.state.service = PredictionService.from_artifact(model_path)
            app.state.load_error = None
            logger.info(
                "loaded %s model for %s",
                app.state.service.metadata.representation,
                app.state.service.metadata.target,
            )
        except (FileNotFoundError, ValueError) as error:
            # Logged at ERROR and re-reported through /health rather than
            # swallowed: an unready service that looks healthy is worse than
            # one that refuses to start.
            app.state.service = None
            app.state.load_error = str(error)
            logger.error("no model loaded: %s", error)

        yield

        app.state.service = None

    app = FastAPI(
        title="Molecular Property Predictor",
        description=DESCRIPTION,
        version="0.1.0",
        lifespan=lifespan,
    )
    _register_routes(app)
    return app


def get_service(request: Request) -> PredictionService:
    """Return the loaded service, or fail the request with a 503.

    Declared as a FastAPI dependency so the check sits in one place instead of
    at the top of every endpoint. 503 rather than 500: the request was
    well-formed and the service is temporarily unable to answer it, which is a
    different thing from the service being broken by it.
    """
    service = getattr(request.app.state, "service", None)
    if service is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=(
                getattr(request.app.state, "load_error", None)
                or "no model is loaded"
            ),
        )
    return service


def _register_routes(app: FastAPI) -> None:
    """Attach the endpoints to ``app``."""

    @app.get("/", summary="Where to go next", tags=["meta"])
    def root() -> dict[str, str]:
        """A deployed URL opened in a browser should not return 404."""
        return {
            "service": "molecular-property-predictor",
            "docs": "/docs",
            "health": "/health",
            "model": "/model",
        }

    @app.get("/health", response_model=Health, tags=["meta"])
    def health(request: Request) -> Health:
        """Liveness, plus whether a model is actually loaded.

        Deliberately does not depend on ``get_service``: this endpoint has to
        answer *especially* when the model is missing, which is the case it
        exists to report.
        """
        return Health(
            status="ok",
            model_loaded=getattr(request.app.state, "service", None) is not None,
        )

    @app.get("/model", response_model=ModelInfo, tags=["meta"])
    def model_info(service: PredictionService = Depends(get_service)) -> ModelInfo:
        """The provenance of the checkpoint being served.

        Reported from the artifact itself rather than from a constant, so it
        cannot drift away from what is actually answering requests.
        """
        metadata = service.metadata
        return ModelInfo(
            target=metadata.target,
            units="eV",
            representation=metadata.representation,
            n_features=metadata.n_features,
            n_parameters=sum(p.numel() for p in service.model.parameters()),
            hidden_sizes=list(metadata.hidden_sizes),
            dropout=metadata.dropout,
            loss_name=metadata.loss_name,
            split_seed=metadata.split_seed,
            torch_seed=metadata.torch_seed,
            epochs_trained=metadata.epochs_trained,
            best_epoch=metadata.best_epoch,
            validation_mae_ev=metadata.validation_mae_ev,
            test_mae_ev=metadata.test_mae_ev,
        )

    @app.post("/predict", response_model=Prediction, tags=["prediction"])
    def predict_one(
        molecule: Molecule, service: PredictionService = Depends(get_service)
    ) -> Prediction:
        """Predict the target property for a single molecule.

        A plain ``def``: see the module docstring on why an ``async def`` here
        would stall every concurrent request behind this one.
        """
        value = service.predict_one(molecule.to_arrays())
        return Prediction(lumo_ev=value, n_atoms=len(molecule.atomic_numbers))

    @app.post("/predict/batch", response_model=BatchPrediction, tags=["prediction"])
    def predict_batch(
        batch: BatchRequest, service: PredictionService = Depends(get_service)
    ) -> BatchPrediction:
        """Predict for many molecules in one pass, in the order submitted.

        One forward pass over the whole batch rather than a loop over
        ``predict_one``: the per-call overhead dominates for a network this
        small, which is the entire reason this endpoint exists.
        """
        values = service.predict([molecule.to_arrays() for molecule in batch.molecules])
        return BatchPrediction(
            predictions=[
                Prediction(lumo_ev=float(value), n_atoms=len(molecule.atomic_numbers))
                for value, molecule in zip(values, batch.molecules, strict=True)
            ]
        )


#: The application uvicorn serves:
#: ``uvicorn molecular_property_predictor.api.main:app``
app = create_app()

"""Featurise and predict. Deliberately knows nothing about HTTP.

**Why this is a separate module from the endpoints.** Everything here can be
exercised without a client, a port, or a running server -- the tests construct
one of these directly and call it. That is the same seam Phase 5 cut when
``train_model`` took an ``on_epoch`` callback instead of importing MLflow: the
part that does the work should not depend on the part that reports it. Here it
buys three things. The prediction path is testable in isolation, the HTTP layer
stays thin enough to read in one sitting, and swapping FastAPI for anything else
would not touch this file.

**Why the model is loaded once and held.** :class:`PredictionService` is built
at application start-up and reused for the lifetime of the process. Loading a
checkpoint takes tens of milliseconds and inference on one molecule takes well
under one; doing the load per request would make the load essentially the entire
response time, for no benefit, on every single call. This is the most common
mistake in naive model serving and it is invisible until someone measures.

**Why the featuriser is chosen by the artifact, not hardcoded.** The checkpoint
records which representation it was trained on, so the service reads that and
looks up the matching function. Hardcoding ``sorted_coulomb_features`` would
work today and would silently serve wrong numbers the moment a checkpoint
trained on the eigenspectrum was pointed at it -- the vector lengths differ, so
it would at least raise, but the failure would come from a shape mismatch deep
in the network rather than from the place that made the wrong assumption.
"""

from __future__ import annotations

import os
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from molecular_property_predictor.features import FEATURE_KINDS
from molecular_property_predictor.model import (
    DEFAULT_MODEL_DIR,
    ArtifactMetadata,
    MolecularNet,
    load_artifact,
)
from molecular_property_predictor.model import predict as predict_from_features

#: Environment variable naming the checkpoint to serve.
#:
#: Configuration comes from the environment rather than a constant in the source
#: because the same image has to run in three places that disagree about paths:
#: this laptop, the Phase 7 container, and whatever Phase 8 deploys onto. That
#: convention -- code identical everywhere, configuration injected per
#: environment -- is why the container built in Phase 7 can be the exact artifact
#: that gets deployed, rather than one rebuilt with production values baked in.
MODEL_PATH_ENV = "MODEL_PATH"

#: Default checkpoint, used when :data:`MODEL_PATH_ENV` is unset.
#:
#: A fixed name rather than "newest file in models/" or "best by validation
#: MAE". Which model is served is a decision, made once and recorded, not
#: something rediscovered at start-up from whatever happens to be on disk --
#: otherwise an unrelated training run changes what the API serves.
DEFAULT_ARTIFACT = DEFAULT_MODEL_DIR / "served.pt"

#: Geometry as the featurisers take it: ``(atomic_numbers, coordinates)``.
Geometry = tuple[np.ndarray, np.ndarray]


def resolve_model_path(path: str | Path | None = None) -> Path:
    """Decide which checkpoint to serve: argument, then environment, then default.

    Args:
        path: An explicit override, which wins over everything. Mainly for
            tests, which must never depend on the developer's environment.

    Returns:
        The path to load. Not checked for existence here -- that is
        :meth:`PredictionService.from_artifact`'s job, so that the error names
        the file and says how to produce it.
    """
    if path is not None:
        return Path(path)

    from_environment = os.environ.get(MODEL_PATH_ENV)
    if from_environment:
        return Path(from_environment)

    return DEFAULT_ARTIFACT


@dataclass(frozen=True)
class PredictionService:
    """A loaded model, its scaler, and the featuriser they were trained with.

    Frozen because nothing about a loaded model should change while it is
    serving. Swapping the checkpoint means building a new service and pointing
    the application at it, which is an explicit act rather than a mutation
    happening under concurrent requests.
    """

    model: MolecularNet
    scaler: object
    metadata: ArtifactMetadata

    @classmethod
    def from_artifact(cls, path: str | Path | None = None) -> PredictionService:
        """Load a checkpoint and verify it can actually serve.

        The consistency check is the point of doing this at start-up rather
        than lazily on first request. If the recorded representation produces
        vectors of a different length than the network expects, this artifact
        can never answer anything -- and the honest moment to fail is when the
        process starts, not when the first user asks. A container that starts
        cleanly and then errors on every request is far harder to diagnose than
        one that refuses to start.

        Raises:
            FileNotFoundError: If no checkpoint is at ``path``.
            ValueError: If the artifact is internally inconsistent.
        """
        resolved = resolve_model_path(path)
        if not resolved.exists():
            raise FileNotFoundError(
                f"no model artifact at {resolved}. Set the {MODEL_PATH_ENV} "
                f"environment variable to a checkpoint, or write one to "
                f"{DEFAULT_ARTIFACT}. Artifacts are build outputs and are "
                f"gitignored, so a fresh clone has none until a model is "
                f"trained or copied in."
            )

        model, scaler, metadata = load_artifact(resolved, device="cpu")

        if metadata.representation not in FEATURE_KINDS:
            raise ValueError(
                f"artifact {resolved} was trained on representation "
                f"{metadata.representation!r}, which this build cannot compute; "
                f"expected one of {sorted(FEATURE_KINDS)}"
            )

        service = cls(model=model, scaler=scaler, metadata=metadata)
        service._check_feature_length(resolved)
        return service

    def _check_feature_length(self, path: Path) -> None:
        """Featurise one trivial molecule and confirm the width lines up.

        A single hydrogen atom is enough: every representation here pads to a
        fixed width, so the vector length does not depend on the molecule.
        """
        probe = (np.array([1], dtype=np.int64), np.zeros((1, 3), dtype=np.float64))
        width = len(self.featurizer(*probe))

        if width != self.metadata.n_features:
            raise ValueError(
                f"artifact {path} records representation "
                f"{self.metadata.representation!r} and n_features="
                f"{self.metadata.n_features}, but that representation produces "
                f"{width} features. The checkpoint's metadata does not match "
                f"its weights."
            )

    @property
    def featurizer(self):
        """The function turning ``(atomic_numbers, coordinates)`` into features."""
        return FEATURE_KINDS[self.metadata.representation]

    def featurize(self, geometries: Sequence[Geometry]) -> np.ndarray:
        """Build the feature matrix for a batch of molecules.

        Returns:
            Shape ``(len(geometries), n_features)``, unscaled. Scaling belongs
            to :func:`~molecular_property_predictor.model.predict`, which owns
            the scaler that must accompany these weights.
        """
        return np.asarray(
            [self.featurizer(numbers, coordinates)
             for numbers, coordinates in geometries],
            dtype=np.float64,
        )

    def predict(self, geometries: Sequence[Geometry]) -> np.ndarray:
        """Predict the target property for each molecule, in order.

        Batched rather than one-at-a-time because the per-call overhead of
        moving a tensor to the device is a large fraction of the cost for a
        network this small; the single-molecule endpoint is a batch of one.

        Returns:
            Shape ``(len(geometries),)``, in the units the model was trained in
            -- eV, since :mod:`~molecular_property_predictor.data` converts the
            energetic QM9 properties on load.
        """
        if not geometries:
            return np.empty(0, dtype=np.float32)

        return predict_from_features(
            self.model, self.scaler, self.featurize(geometries), device="cpu"
        )

    def predict_one(self, geometry: Geometry) -> float:
        """Predict for a single molecule, as a plain float.

        Returns a Python float rather than a NumPy scalar so that the value
        serialises to JSON without the response model having to coerce it.
        """
        return float(self.predict([geometry])[0])

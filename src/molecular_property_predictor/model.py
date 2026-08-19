"""The neural network, and the artifact that makes it servable.

Phase 3 earned this phase rather than assuming it: gradient boosting beat ridge
on every representation, so the relationship between geometry and LUMO is
non-linear, and both learning curves had flattened, so more data was not the
answer. That is the case for a model with more capacity.

The architecture is deliberately small and ordinary -- a stack of linear layers
with a non-linearity between them. Nothing here is novel, and it is not meant
to be. The interesting part of Phase 4 is the training loop in
:mod:`~molecular_property_predictor.train`, not the shape of the network.

**On the artifact.** A saved network is useless on its own. Predictions require
exactly the preprocessing the model was trained under, so :func:`save_artifact`
bundles the fitted scaler *with* the weights. Shipping them separately is a
well-known way to end up with a served model that silently returns nonsense --
nothing errors, the numbers are just wrong.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import torch
from sklearn.preprocessing import StandardScaler
from torch import nn

#: Where trained artifacts are written. Build outputs, so gitignored: they are
#: reproducible from the code and the seed.
from molecular_property_predictor.data import PROJECT_ROOT

DEFAULT_MODEL_DIR = PROJECT_ROOT / "models"


def resolve_device(device: str | torch.device | None = None) -> torch.device:
    """Pick a device, preferring CUDA when it is available.

    Training happens on whatever this machine has; *serving* will not. The
    Phase 7 container gets the CPU build of torch, because inference needs no
    GPU and the CUDA wheel is roughly 2.5 GB. That asymmetry -- train on
    expensive hardware as a batch job, serve on cheap hardware as a latency
    sensitive service -- is the normal shape of a production system, and it is
    why :func:`load_artifact` always maps tensors to CPU.
    """
    if device is not None:
        return torch.device(device)
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


class MolecularNet(nn.Module):
    """A feed-forward network mapping a feature vector to one property.

    Args:
        n_features: Length of the input vector (435 for the sorted Coulomb
            matrix, 29 for the eigenspectrum, 5 for composition).
        hidden_sizes: Width of each hidden layer.
        dropout: Fraction of activations zeroed during training only.
            Dropout is regularisation: by randomly removing units it stops the
            network leaning on any single one, which makes it generalise
            better. It is active in ``train()`` mode and disabled in ``eval()``
            mode -- forgetting to switch modes does not raise, it just returns
            noisy predictions.
    """

    def __init__(
        self,
        n_features: int,
        hidden_sizes: tuple[int, ...] = (512, 256, 128),
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.n_features = n_features
        self.hidden_sizes = tuple(hidden_sizes)
        self.dropout = dropout

        layers: list[nn.Module] = []
        in_size = n_features
        for out_size in hidden_sizes:
            layers.append(nn.Linear(in_size, out_size))
            # Without a non-linearity between them, a stack of linear layers
            # collapses algebraically into a single linear layer -- the depth
            # would buy exactly nothing. ReLU is what makes the stack more
            # expressive than the ridge regression of Phase 3.
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout))
            in_size = out_size
        layers.append(nn.Linear(in_size, 1))

        self.layers = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Map a batch of feature vectors to a batch of predictions.

        Returns:
            Shape ``(batch,)`` -- squeezed, so it lines up with the targets
            rather than being ``(batch, 1)``. Mismatching those two shapes is
            a silent bug: broadcasting turns a ``(batch,)`` target and a
            ``(batch, 1)`` prediction into a ``(batch, batch)`` loss, which
            trains, converges to something, and is wrong.
        """
        return self.layers(x).squeeze(-1)


@dataclass(frozen=True)
class ArtifactMetadata:
    """Everything needed to reproduce or interpret a trained model.

    Recorded so that a saved model can be traced back to how it was made. A
    checkpoint whose split seed is unknown cannot be honestly evaluated -- you
    have no way to tell whether a given molecule was in its training set.
    """

    representation: str
    target: str
    n_features: int
    hidden_sizes: tuple[int, ...]
    dropout: float
    loss_name: str
    split_seed: int
    torch_seed: int
    epochs_trained: int
    best_epoch: int
    validation_mae_ev: float
    #: Mean absolute error on the held-out test split, in eV.
    #:
    #: Optional, with a default, for two reasons. The first is compatibility:
    #: ``load_artifact`` reconstructs this dataclass by keyword from whatever
    #: the checkpoint stored, so a required field would make every artifact
    #: written before Phase 6 fail to load.
    #:
    #: The second is honest bookkeeping. Every score in Phases 3 to 5 is a
    #: *validation* score, because the test split is opened exactly once, for
    #: the single checkpoint that ships. ``None`` therefore means "not
    #: measured", which is a different statement from a number -- and a
    #: different statement from zero.
    test_mae_ev: float | None = None


def save_artifact(
    path: Path | str,
    model: MolecularNet,
    scaler: StandardScaler,
    metadata: ArtifactMetadata,
) -> Path:
    """Write weights, scaler and provenance as one file.

    The scaler is stored as plain arrays rather than a pickled scikit-learn
    object. Unpickling executes code and is version-fragile, which is a poor
    property for something a web service loads at start-up; two float arrays
    are neither.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    torch.save(
        {
            "state_dict": model.state_dict(),
            "scaler_mean": np.asarray(scaler.mean_, dtype=np.float64),
            "scaler_scale": np.asarray(scaler.scale_, dtype=np.float64),
            "metadata": asdict(metadata),
        },
        path,
    )
    return path


def load_artifact(
    path: Path | str, device: str | torch.device | None = None
) -> tuple[MolecularNet, StandardScaler, ArtifactMetadata]:
    """Read back what :func:`save_artifact` wrote.

    Tensors are loaded with ``map_location="cpu"`` first, so an artifact
    trained on a GPU opens on a machine that has none. Without that, Phase 6
    would break the first time the API ran anywhere but this laptop.

    The returned model is in ``eval()`` mode: dropout off, which is what
    inference requires.
    """
    checkpoint = torch.load(Path(path), map_location="cpu", weights_only=False)
    metadata = ArtifactMetadata(**checkpoint["metadata"])

    model = MolecularNet(
        n_features=metadata.n_features,
        hidden_sizes=tuple(metadata.hidden_sizes),
        dropout=metadata.dropout,
    )
    model.load_state_dict(checkpoint["state_dict"])
    model.to(resolve_device(device))
    model.eval()

    # Rebuild the scaler from its arrays. `n_features_in_` and `var_` are set
    # because scikit-learn's transform() expects a fitted estimator's full
    # attribute set, not just the two arrays it actually uses.
    scaler = StandardScaler()
    scaler.mean_ = checkpoint["scaler_mean"]
    scaler.scale_ = checkpoint["scaler_scale"]
    scaler.var_ = scaler.scale_**2
    scaler.n_features_in_ = len(scaler.mean_)

    return model, scaler, metadata


def predict(
    model: MolecularNet,
    scaler: StandardScaler,
    features: np.ndarray,
    *,
    device: str | torch.device | None = None,
    batch_size: int = 4096,
) -> np.ndarray:
    """Run raw (unscaled) features through scaler and network.

    Takes raw features rather than scaled ones on purpose: the caller should
    never have to remember to scale first, because forgetting is silent. This
    is the function Phase 6 will sit behind an HTTP endpoint.
    """
    resolved = resolve_device(device)
    model = model.to(resolved)
    # eval() disables dropout; no_grad() stops autograd recording a graph we
    # would never call backward() on, which saves both time and memory.
    model.eval()

    scaled = scaler.transform(np.asarray(features, dtype=np.float64))
    predictions = []
    with torch.no_grad():
        for start in range(0, len(scaled), batch_size):
            batch = torch.as_tensor(
                scaled[start : start + batch_size], dtype=torch.float32
            ).to(resolved)
            predictions.append(model(batch).cpu().numpy())

    return np.concatenate(predictions) if predictions else np.empty(0)

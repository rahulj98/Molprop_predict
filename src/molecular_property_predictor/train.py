"""The training loop, written out rather than hidden behind a wrapper.

PyTorch Lightning or skorch would collapse most of this file to ten lines, and
would also hide the exact thing this phase exists to understand. The loop is
five lines of real work::

    predictions = model(x_batch)          # 1. forward pass
    loss = loss_fn(predictions, y_batch)  # 2. how wrong was it
    optimizer.zero_grad()                 # 3. clear the last step's gradients
    loss.backward()                       # 4. backward pass -- autograd
    optimizer.step()                      # 5. nudge every weight

1. **Forward pass.** Data flows through the layers to a prediction. Matrix
   multiplies and a non-linearity; nothing more mysterious than that.

2. **Loss.** One scalar summarising how wrong the batch was. Everything after
   this exists to make that number smaller.

3. **zero_grad.** PyTorch *accumulates* into ``.grad`` rather than overwriting
   it. Omit this line and every step is polluted by every step before it -- and
   nothing raises, the model just trains badly. The default looks like a trap
   but is deliberate: it is what lets you split one large batch into several
   small forward/backward passes and sum their gradients, which is how people
   train models too big to fit a full batch in memory.

4. **backward.** During the forward pass autograd recorded a graph of every
   operation. ``backward()`` walks it in reverse and applies the chain rule to
   get d(loss)/d(parameter) for every parameter at once. This is the part worth
   genuinely absorbing: you never write a derivative, because the graph already
   knows them.

5. **step.** The optimiser turns those gradients into an actual update. Adam
   rather than plain SGD because it keeps a per-parameter adaptive step size,
   which matters when different inputs have very different scales and
   curvatures -- exactly our case, even after standardisation.
"""

from __future__ import annotations

import copy
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.preprocessing import StandardScaler
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from molecular_property_predictor.baseline import evaluate, split_positions
from molecular_property_predictor.data import (
    DEFAULT_PROCESSED_DIR,
    TARGET_PROPERTY,
)
from molecular_property_predictor.features import load_features
from molecular_property_predictor.model import (
    DEFAULT_MODEL_DIR,
    ArtifactMetadata,
    MolecularNet,
    resolve_device,
    save_artifact,
)

#: Loss functions to choose between.
#:
#: This is a more interesting choice than it looks. Phase 3 found the best
#: baseline had a predicted-vs-true slope of 0.811 -- it shrank towards the
#: mean. That is partly what squared error *asks* for: the minimiser of MSE is
#: the conditional mean, so it hedges towards the centre wherever it is
#: uncertain, and it weights a 2 eV miss sixteen times as heavily as a 0.5 eV
#: one. The minimiser of L1 is the conditional median instead.
#:
#: Our reported metric is MAE, so training on MSE optimises something we do not
#: report. Huber is the compromise: quadratic near zero (smooth gradients where
#: it matters) and linear in the tails (no outlier domination). Phase 4
#: measures whether the mismatch is worth caring about rather than assuming it.
LOSSES: dict[str, Callable[[], nn.Module]] = {
    "mse": nn.MSELoss,
    "huber": lambda: nn.HuberLoss(delta=1.0),
    "l1": nn.L1Loss,
}


@dataclass(frozen=True)
class TrainingConfig:
    """Everything that determines a training run."""

    representation: str = "sorted_coulomb"
    hidden_sizes: tuple[int, ...] = (512, 256, 128)
    dropout: float = 0.1
    loss_name: str = "mse"
    learning_rate: float = 1e-3
    batch_size: int = 256
    max_epochs: int = 100
    #: Stop after this many epochs with no improvement in validation MAE.
    patience: int = 10
    split_seed: int = 0
    torch_seed: int = 0


@dataclass
class TrainingRun:
    """The result of :func:`train_model`."""

    model: MolecularNet
    scaler: StandardScaler
    config: TrainingConfig
    history: pd.DataFrame
    best_epoch: int
    metrics: dict[str, float]
    seconds: float
    device: str
    extras: dict = field(default_factory=dict)

    def metadata(self) -> ArtifactMetadata:
        return ArtifactMetadata(
            representation=self.config.representation,
            target=self.extras.get("target", TARGET_PROPERTY),
            n_features=self.model.n_features,
            hidden_sizes=self.model.hidden_sizes,
            dropout=self.model.dropout,
            loss_name=self.config.loss_name,
            split_seed=self.config.split_seed,
            torch_seed=self.config.torch_seed,
            epochs_trained=len(self.history),
            best_epoch=self.best_epoch,
            validation_mae_ev=self.metrics["mae_ev"],
        )


def set_seed(seed: int) -> None:
    """Seed Python-side and CUDA-side generators.

    Honest caveat: this makes runs reproducible in practice but not bitwise
    guaranteed on a GPU. Some cuDNN kernels pick algorithms based on runtime
    heuristics and accumulate floating-point sums in a non-deterministic order.
    Forcing full determinism costs speed, so we take reproducible-in-practice
    and say so rather than claiming more than we have.
    """
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def prepare_data(
    frame: pd.DataFrame,
    config: TrainingConfig,
    *,
    target: str = TARGET_PROPERTY,
    processed_dir: Path | str = DEFAULT_PROCESSED_DIR,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, StandardScaler]:
    """Load features, split, and scale using training statistics only.

    The same discipline as Phase 3, and for the same reason: a scaler fitted on
    everything mixes validation statistics into training, nothing raises, and
    every score comes out quietly better than the truth. Here it has to be done
    by hand rather than by a scikit-learn ``Pipeline``, which makes the
    assertion in ``tests/test_train.py`` more valuable, not less.
    """
    train_rows, validation_rows, _ = split_positions(frame, seed=config.split_seed)
    features = load_features(frame, config.representation, processed_dir)
    y = frame[target].to_numpy(dtype=np.float32)

    scaler = StandardScaler().fit(features[train_rows])
    x_train = scaler.transform(features[train_rows]).astype(np.float32)
    x_validation = scaler.transform(features[validation_rows]).astype(np.float32)

    return x_train, y[train_rows], x_validation, y[validation_rows], scaler


@torch.no_grad()
def evaluate_model(
    model: MolecularNet, x: torch.Tensor, y: np.ndarray, batch_size: int = 8192
) -> dict[str, float]:
    """Score a model without letting autograd record anything.

    ``eval()`` disables dropout -- scoring with dropout on measures a different,
    noisier model than the one being served. ``no_grad()`` stops the graph being
    built at all, since nothing here will ever call ``backward()``.
    """
    model.eval()
    predictions = [
        model(x[start : start + batch_size]).cpu().numpy()
        for start in range(0, len(x), batch_size)
    ]
    return evaluate(y, np.concatenate(predictions))


def train_model(
    frame: pd.DataFrame,
    config: TrainingConfig | None = None,
    *,
    target: str = TARGET_PROPERTY,
    processed_dir: Path | str = DEFAULT_PROCESSED_DIR,
    device: str | torch.device | None = None,
    report: Callable[[str], None] | None = None,
    on_epoch: Callable[[int, dict[str, float]], None] | None = None,
) -> TrainingRun:
    """Train one network and return the best epoch's weights.

    Early stopping keeps whatever scored best on **validation**, not whatever
    the last epoch happened to produce -- past a certain point the training loss
    keeps falling while validation stops improving, and the final weights are
    then worse than an earlier snapshot. Note that this is another choice made
    by looking at validation, which is exactly why the test split stays sealed.

    Args:
        on_epoch: Called after every epoch with ``(epoch, metrics)``, where
            ``metrics`` holds the training loss and the validation scores.
            This is the seam Phase 5 hangs experiment tracking on. It is a
            plain callback rather than an MLflow call inlined here on purpose:
            this module should train networks and know nothing about where the
            numbers are being recorded, so the tracking backend can be swapped
            or dropped without touching the loop.
        report: Called with preformatted lines, for humans watching a notebook.

    Returns:
        A :class:`TrainingRun`; ``model`` already carries the best weights.
    """
    config = config or TrainingConfig()
    if config.loss_name not in LOSSES:
        raise ValueError(
            f"unknown loss {config.loss_name!r}; expected one of {sorted(LOSSES)}"
        )

    resolved = resolve_device(device)
    set_seed(config.torch_seed)

    x_train, y_train, x_validation, y_validation, scaler = prepare_data(
        frame, config, target=target, processed_dir=processed_dir
    )

    # The whole training set is ~180 MB as float32, so it fits in GPU memory.
    # Moving it once up front rather than per batch removes a host-to-device
    # copy from the inner loop, which would otherwise dominate the runtime of a
    # network this small.
    x_train_t = torch.as_tensor(x_train).to(resolved)
    y_train_t = torch.as_tensor(y_train).to(resolved)
    x_validation_t = torch.as_tensor(x_validation).to(resolved)

    # DataLoader here is doing batching and per-epoch shuffling, not streaming:
    # the data already fits in memory. num_workers=0 because the tensors are
    # already on the device, so worker processes would only add overhead.
    loader = DataLoader(
        TensorDataset(x_train_t, y_train_t),
        batch_size=config.batch_size,
        shuffle=True,
        num_workers=0,
    )

    model = MolecularNet(
        n_features=x_train.shape[1],
        hidden_sizes=config.hidden_sizes,
        dropout=config.dropout,
    ).to(resolved)
    loss_fn = LOSSES[config.loss_name]()
    optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate)

    history: list[dict] = []
    best_mae = float("inf")
    best_state: dict | None = None
    best_epoch = 0
    epochs_without_improvement = 0
    start_time = time.perf_counter()

    for epoch in range(1, config.max_epochs + 1):
        # train() enables dropout. Forgetting this after an eval pass leaves
        # the network in inference mode and quietly removes the regularisation.
        model.train()
        running_loss = 0.0
        for x_batch, y_batch in loader:
            predictions = model(x_batch)         # 1. forward pass
            loss = loss_fn(predictions, y_batch)  # 2. how wrong was it
            optimizer.zero_grad()                 # 3. clear old gradients
            loss.backward()                       # 4. backward pass
            optimizer.step()                      # 5. update the weights
            running_loss += loss.item() * len(x_batch)

        train_loss = running_loss / len(x_train)
        metrics = evaluate_model(model, x_validation_t, y_validation)
        history.append({"epoch": epoch, "train_loss": train_loss, **metrics})

        if on_epoch is not None:
            on_epoch(epoch, {"train_loss": train_loss, **metrics})

        if metrics["mae_ev"] < best_mae:
            best_mae = metrics["mae_ev"]
            # deepcopy, not a reference: state_dict() returns live tensors that
            # the next optimizer.step() would overwrite in place.
            best_state = copy.deepcopy(model.state_dict())
            best_epoch = epoch
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1

        if report is not None:
            report(
                f"epoch {epoch:>3}  train_loss {train_loss:.4f}  "
                f"val MAE {metrics['mae_ev']:.4f} eV  "
                f"val R2 {metrics['r2']:.4f}"
                f"{'  *' if epoch == best_epoch else ''}"
            )

        if epochs_without_improvement >= config.patience:
            if report is not None:
                report(
                    f"early stop: no improvement for {config.patience} epochs "
                    f"(best was epoch {best_epoch})"
                )
            break

    assert best_state is not None  # at least one epoch always runs
    model.load_state_dict(best_state)

    return TrainingRun(
        model=model,
        scaler=scaler,
        config=config,
        history=pd.DataFrame(history),
        best_epoch=best_epoch,
        metrics=evaluate_model(model, x_validation_t, y_validation),
        seconds=time.perf_counter() - start_time,
        device=str(resolved),
        extras={"target": target},
    )


def train_and_cache(
    frame: pd.DataFrame,
    config: TrainingConfig,
    name: str,
    *,
    target: str = TARGET_PROPERTY,
    model_dir: Path | str = DEFAULT_MODEL_DIR,
    processed_dir: Path | str = DEFAULT_PROCESSED_DIR,
    use_cache: bool = True,
    report: Callable[[str], None] | None = None,
) -> tuple[Path, pd.DataFrame]:
    """Train once, then reuse the artifact and its per-epoch history.

    Training this network takes roughly twenty minutes on CPU, which a notebook
    must not repeat on every execution. Both outputs are written under
    ``model_dir`` and are gitignored: they are build products, reproducible
    from the code plus the seeds recorded in the artifact's metadata.

    Returns:
        ``(artifact_path, history)``.
    """
    model_dir = Path(model_dir)
    artifact = model_dir / f"molecular_net_{name}.pt"
    history_path = model_dir / f"history_{name}.csv"

    if use_cache and artifact.exists() and history_path.exists():
        return artifact, pd.read_csv(history_path)

    run = train_model(
        frame,
        config,
        target=target,
        processed_dir=processed_dir,
        report=report,
    )
    save_artifact(artifact, run.model, run.scaler, run.metadata())
    model_dir.mkdir(parents=True, exist_ok=True)
    run.history.to_csv(history_path, index=False)
    return artifact, run.history

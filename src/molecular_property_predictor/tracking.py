"""Experiment tracking: a queryable record of every training run.

Phase 4 compared three runs by holding their results in a Python dict and
printing a table. That is fine for three runs in one sitting, and it fails in a
specific, unglamorous way as soon as there are more:

- ``models/history_mse.csv`` records what happened but not *what was asked for*.
  Nothing on disk connects that file to ``learning_rate=1e-3``.
- Re-running the notebook with a different width silently overwrites both the
  artifact and its history.
- The comparison table lives in a kernel that will be restarted.

**What MLflow actually is.** A database of training runs, plus a UI over it.
Not a training framework -- it does not touch the loop, and the loop below
still knows nothing about it (see the ``on_epoch`` callback in
:mod:`~molecular_property_predictor.train`). Each *run* records:

- **params**: the inputs, written once, immutable. Learning rate, width, seed.
- **metrics**: the outputs, each a time series indexed by step. Loss per epoch.
- **tags**: free-form labels for grouping, e.g. which sweep a run belonged to.
- **artifacts**: files attached to the run -- our checkpoint, its history.

The params/metrics distinction is the one that matters, and it is not
bureaucratic: it is what lets the UI plot twelve validation curves on one axis
and colour them by learning rate.

**Why not just a CSV?** You can get a long way with one, and for nine runs it
would work. It stops working when two runs write concurrently, when you want
"every run with dropout > 0.1 sorted by validation MAE", or when you need the
checkpoint that a given row refers to. Those three are the whole job.

**Storage choice: a local SQLite file.** ``mlruns/mlflow.db`` for the run
records, ``mlruns/artifacts/`` for the files they point at. No server process;
``mlflow ui --backend-store-uri sqlite:///mlruns/mlflow.db`` reads it directly.

The obvious cheaper option -- MLflow's plain-directory "file store", one folder
per run -- is what this project originally reached for, and MLflow 3.x refuses
it: the backend is in maintenance mode and raises unless an opt-out environment
variable is set. The reason is the same one that makes a CSV a poor run log. A
directory tree has no transactions and no query engine, so concurrent writes
race and every filter means walking every run. SQLite is still a single local
file with nothing to administer, and it gives both.

Note the split between the two: run *records* go to the database, run
*artifacts* (our checkpoints) go to a directory, because a 4 MB blob does not
belong in a metadata store. That split is not a local quirk -- it is the same
one a production deployment makes, with Postgres and object storage standing in
for the two halves. The production shape adds a tracking *server* in front so a
team writes to one place, which is worth knowing and not worth running here for
one person on one laptop.

**Deliberately not used: autologging.** ``mlflow.pytorch.autolog()`` is one line
and would capture most of this. Everything below is about eight lines of
explicit calls instead, on the grounds that a run record you cannot account for
is worse than no run record.

**Deliberately not used: the MLflow model registry / model format.**
:func:`mlflow.pytorch.log_model` would store the network in MLflow's own
packaging with a ``python_function`` flavour. Our ``.pt`` from Phase 4 already
bundles the weights with the scaler that must accompany them, so this logs that
file as a plain artifact and records the input signature alongside it. The
consequence is the point: Phase 6's API loads a checkpoint with
:func:`~molecular_property_predictor.model.load_artifact` and never imports
MLflow. Tracking is a development-time concern; the served container should not
inherit it.
"""

from __future__ import annotations

import itertools
from collections.abc import Callable, Iterable, Sequence
from dataclasses import asdict, replace
from pathlib import Path

import mlflow
import numpy as np
import pandas as pd

from molecular_property_predictor.data import (
    DEFAULT_PROCESSED_DIR,
    PROJECT_ROOT,
    TARGET_PROPERTY,
)
from molecular_property_predictor.model import (
    DEFAULT_MODEL_DIR,
    load_artifact,
    save_artifact,
)
from molecular_property_predictor.train import (
    TrainingConfig,
    TrainingRun,
    prepare_data,
    train_model,
)

#: Holds both halves of the local store: ``mlflow.db`` and ``artifacts/``.
#: Gitignored -- it is a record of local runs, largely binary, and committing it
#: would turn every experiment into a commit.
DEFAULT_TRACKING_DIR = PROJECT_ROOT / "mlruns"

#: Runs are grouped into experiments. One per prediction task, so that a later
#: phase targeting a different property does not have to share a leaderboard
#: with this one -- the MAEs would not be comparable.
DEFAULT_EXPERIMENT = "qm9-lumo"


def tracking_uri(directory: Path | str = DEFAULT_TRACKING_DIR) -> str:
    """Return the SQLite URI for the run database under ``directory``.

    A URI rather than a path because the same setting selects Postgres or an
    HTTP tracking server in a deployed setup; the scheme is how MLflow tells
    them apart, so this one line is the whole difference between local and
    remote tracking. Absolute, for the reason the data paths are: a notebook
    running from ``notebooks/`` must not quietly start a second store.

    ``as_posix`` because SQLAlchemy wants forward slashes even on Windows.
    """
    return f"sqlite:///{Path(directory).resolve().as_posix()}/mlflow.db"


def artifact_uri(directory: Path | str = DEFAULT_TRACKING_DIR) -> str:
    """Return the URI of the directory that holds run artifacts.

    Set explicitly because MLflow's default is ``./mlartifacts`` *relative to
    the working directory*, which would scatter checkpoints across the repo
    depending on where each run was launched from.
    """
    path = Path(directory).resolve() / "artifacts"
    path.mkdir(parents=True, exist_ok=True)
    return path.as_uri()


def configure(
    directory: Path | str = DEFAULT_TRACKING_DIR,
    experiment: str = DEFAULT_EXPERIMENT,
) -> str:
    """Point MLflow at the local store and select the experiment.

    Both settings are process-global state in MLflow, which is worth knowing
    rather than discovering: every later ``start_run`` in this interpreter
    lands wherever this call pointed it.

    The experiment is created explicitly on first use rather than left to
    ``set_experiment``, because only :func:`mlflow.create_experiment` accepts an
    artifact location, and an experiment's artifact root is fixed at creation.

    Returns:
        The experiment id.
    """
    Path(directory).mkdir(parents=True, exist_ok=True)
    mlflow.set_tracking_uri(tracking_uri(directory))

    if mlflow.get_experiment_by_name(experiment) is None:
        mlflow.create_experiment(experiment, artifact_location=artifact_uri(directory))

    return mlflow.set_experiment(experiment).experiment_id


def config_params(config: TrainingConfig) -> dict[str, str]:
    """Flatten a :class:`TrainingConfig` into loggable params.

    Everything is stringified because that is how MLflow stores params, and
    doing it here rather than relying on implicit conversion keeps
    ``hidden_sizes`` readable as ``(512, 256, 128)`` instead of however a
    tuple's repr happens to land.
    """
    return {key: str(value) for key, value in asdict(config).items()}


def _signature_dict(n_features: int) -> dict:
    """Describe the input and output arrays this model expects.

    A signature is a shape-and-dtype contract. Logging it means the run record
    itself states that this checkpoint consumes ``(n, n_features)`` float32 and
    returns ``(n,)`` float32 -- which is exactly what Phase 6 has to validate
    incoming requests against. Without it, the only place that contract exists
    is in the head of whoever trained the model.
    """
    example_x = np.zeros((1, n_features), dtype=np.float32)
    example_y = np.zeros(1, dtype=np.float32)
    return mlflow.models.infer_signature(example_x, example_y).to_dict()


def track_training(
    frame: pd.DataFrame,
    config: TrainingConfig,
    *,
    run_name: str,
    target: str = TARGET_PROPERTY,
    tags: dict[str, str] | None = None,
    processed_dir: Path | str = DEFAULT_PROCESSED_DIR,
    model_dir: Path | str = DEFAULT_MODEL_DIR,
    device: str | None = None,
    report: Callable[[str], None] | None = None,
) -> tuple[str, TrainingRun]:
    """Train one network inside an MLflow run, recording everything about it.

    The training loop is untouched: it is handed an ``on_epoch`` callback and
    remains unaware of what that callback does. Swapping MLflow for Weights &
    Biases, or for nothing at all, would not change a line of
    :mod:`~molecular_property_predictor.train`.

    What gets recorded, and why each part is needed to make the run
    reproducible after the fact:

    - **params**: the config, plus split sizes and parameter count. Without the
      seeds, a run cannot be repeated; without the split sizes, its MAE cannot
      be compared to another run's.
    - **metrics, stepped by epoch**: the learning curve, not just its endpoint.
      A run that reached 0.25 eV in ten epochs and one that took eighty are
      different results even though the final number matches.
    - **artifacts**: the ``.pt`` checkpoint, the history CSV, the signature.
      This is the link ``models/`` loses -- the weights and the config that
      produced them, stored together.

    Returns:
        ``(run_id, TrainingRun)``. The run id is the durable handle: it is what
        :func:`load_run_artifact` takes, and it survives the kernel.
    """
    model_dir = Path(model_dir)

    with mlflow.start_run(run_name=run_name) as active:
        mlflow.set_tags({"phase": "5", "target": target, **(tags or {})})
        mlflow.log_params(config_params(config))

        run = train_model(
            frame,
            config,
            target=target,
            processed_dir=processed_dir,
            device=device,
            report=report,
            # The seam. One line, and the loop stays a training loop.
            on_epoch=lambda epoch, metrics: mlflow.log_metrics(metrics, step=epoch),
        )

        x_train, _, x_validation, _, _ = prepare_data(
            frame, config, target=target, processed_dir=processed_dir
        )
        mlflow.log_params(
            {
                "target": target,
                "n_features": x_train.shape[1],
                "n_train": len(x_train),
                "n_validation": len(x_validation),
                "n_parameters": sum(p.numel() for p in run.model.parameters()),
                "device": run.device,
            }
        )

        # Summary metrics without a step: these describe the run as a whole,
        # and are what `search_runs` sorts and filters on. The per-epoch series
        # above is for plotting; this is for comparing.
        mlflow.log_metrics(
            {
                "best_val_mae_ev": run.metrics["mae_ev"],
                "best_val_rmse_ev": run.metrics["rmse_ev"],
                "best_val_r2": run.metrics["r2"],
                "best_epoch": float(run.best_epoch),
                "epochs_trained": float(len(run.history)),
                "train_seconds": run.seconds,
            }
        )

        artifact = save_artifact(
            model_dir / f"molecular_net_{run_name}.pt",
            run.model,
            run.scaler,
            run.metadata(),
        )
        history_path = model_dir / f"history_{run_name}.csv"
        run.history.to_csv(history_path, index=False)

        mlflow.log_artifact(str(artifact))
        mlflow.log_artifact(str(history_path))
        mlflow.log_dict(_signature_dict(x_train.shape[1]), "signature.json")

        return active.info.run_id, run


def make_grid(base: TrainingConfig, **options: Sequence) -> list[TrainingConfig]:
    """Expand a base config into every combination of the given options.

    ``make_grid(base, learning_rate=(1e-2, 1e-3), dropout=(0.0, 0.1))`` gives
    four configs. A full grid rather than a random search because with two or
    three axes and a handful of values each, exhaustive is both cheap and
    easier to reason about; random search wins once the space is large enough
    that most axes turn out not to matter.

    Raises:
        TypeError: If an option is not a field of :class:`TrainingConfig`.
            ``replace`` would raise anyway, but only after the first config had
            already been built and possibly trained.
    """
    valid = set(asdict(base))
    unknown = set(options) - valid
    if unknown:
        raise TypeError(f"not TrainingConfig fields: {sorted(unknown)}")

    names = list(options)
    return [
        replace(base, **dict(zip(names, values, strict=True)))
        for values in itertools.product(*(options[name] for name in names))
    ]


def name_config(config: TrainingConfig, varied: Iterable[str]) -> str:
    """Build a short run name from whichever fields the sweep varies.

    Naming runs after their differences rather than ``run_1 ... run_9`` is a
    small thing that pays for itself the moment you are reading the UI and
    trying to remember which is which.
    """
    parts = []
    for field in varied:
        value = getattr(config, field)
        if isinstance(value, tuple):
            value = "x".join(str(v) for v in value)
        parts.append(f"{field[:2]}{value}")
    return "-".join(parts) or "run"


def sweep(
    frame: pd.DataFrame,
    configs: Sequence[TrainingConfig],
    *,
    varied: Iterable[str] = (),
    sweep_name: str = "sweep",
    target: str = TARGET_PROPERTY,
    processed_dir: Path | str = DEFAULT_PROCESSED_DIR,
    model_dir: Path | str = DEFAULT_MODEL_DIR,
    device: str | None = None,
    report: Callable[[str], None] | None = None,
) -> pd.DataFrame:
    """Train every config, each as its own tracked run.

    Runs are tagged with ``sweep_name`` so the group can be pulled back out
    later. Scores are validation scores throughout -- picking a learning rate
    by looking at these numbers is precisely the kind of choice that spends a
    split's ability to estimate unseen performance, and it is why the test
    split is still sealed going into Phase 6.

    Returns:
        One row per run, sorted by validation MAE, with the run ids.
    """
    varied = list(varied)
    rows = []
    for position, config in enumerate(configs, start=1):
        run_name = f"{sweep_name}-{name_config(config, varied)}"
        if report is not None:
            report(f"[{position}/{len(configs)}] {run_name}")

        run_id, run = track_training(
            frame,
            config,
            run_name=run_name,
            target=target,
            tags={"sweep": sweep_name},
            processed_dir=processed_dir,
            model_dir=model_dir,
            device=device,
        )

        rows.append(
            {
                "run_name": run_name,
                "run_id": run_id,
                **{field: getattr(config, field) for field in varied},
                "val_mae_ev": run.metrics["mae_ev"],
                "val_r2": run.metrics["r2"],
                "best_epoch": run.best_epoch,
                "epochs": len(run.history),
                "seconds": round(run.seconds, 1),
            }
        )
        if report is not None:
            report(
                f"    MAE {run.metrics['mae_ev']:.4f} eV  "
                f"R2 {run.metrics['r2']:.4f}  ({run.seconds:.0f}s)"
            )

    return pd.DataFrame(rows).sort_values("val_mae_ev", ignore_index=True)


def search_runs(
    *,
    experiment: str = DEFAULT_EXPERIMENT,
    directory: Path | str = DEFAULT_TRACKING_DIR,
    filter_string: str = "",
    order_by: Sequence[str] = ("metrics.best_val_mae_ev ASC",),
    max_results: int = 200,
) -> pd.DataFrame:
    """Query the tracking store, returning runs as a DataFrame.

    This is the payoff, and the thing a printed table cannot do: the record
    outlives the process that wrote it and can be filtered after the fact.
    ``filter_string`` uses MLflow's own small query language, e.g.
    ``"params.loss_name = 'mse' and metrics.best_val_mae_ev < 0.3"``.

    Returns:
        Empty frame if the experiment does not exist yet, rather than raising --
        a fresh clone with no runs is a normal state, not an error.
    """
    mlflow.set_tracking_uri(tracking_uri(directory))
    found = mlflow.get_experiment_by_name(experiment)
    if found is None:
        return pd.DataFrame()

    return mlflow.search_runs(
        experiment_ids=[found.experiment_id],
        filter_string=filter_string,
        order_by=list(order_by),
        max_results=max_results,
    )


def best_run(
    *,
    experiment: str = DEFAULT_EXPERIMENT,
    directory: Path | str = DEFAULT_TRACKING_DIR,
    filter_string: str = "",
    metric: str = "metrics.best_val_mae_ev",
) -> pd.Series:
    """The run with the lowest value of ``metric``.

    Raises:
        LookupError: If no run matches. Returning ``None`` here would push the
            failure into whatever tried to read ``.run_id`` off it.
    """
    runs = search_runs(
        experiment=experiment, directory=directory, filter_string=filter_string
    )
    if runs.empty or metric not in runs:
        raise LookupError(f"no runs with {metric} in experiment {experiment!r}")

    return runs.loc[runs[metric].idxmin()]


def load_run_artifact(
    run_id: str,
    *,
    directory: Path | str = DEFAULT_TRACKING_DIR,
    device: str | None = None,
):
    """Download a run's checkpoint and load it.

    Closes the loop that Phase 4 left open: from a row in a results table, back
    to the exact weights that produced it, without having to guess which file
    in ``models/`` is the right one.

    Note that the loading itself is still
    :func:`~molecular_property_predictor.model.load_artifact`. MLflow is only
    locating the file. That separation is what keeps Phase 6 free of it.

    Returns:
        ``(model, scaler, metadata)``.
    """
    mlflow.set_tracking_uri(tracking_uri(directory))
    client = mlflow.tracking.MlflowClient()

    checkpoints = [
        item.path
        for item in client.list_artifacts(run_id)
        if item.path.endswith(".pt")
    ]
    if not checkpoints:
        raise LookupError(f"run {run_id} has no .pt artifact")

    local = mlflow.artifacts.download_artifacts(
        run_id=run_id, artifact_path=checkpoints[0]
    )
    return load_artifact(local, device=device)
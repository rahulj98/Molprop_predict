"""Measure what the hand-written batching loop saves against ``DataLoader``.

``train.shuffled_batches`` replaces ``DataLoader(TensorDataset(...),
shuffle=True)``, and its docstring claims a specific saving. This script is
what that number comes from, committed so the claim can be re-checked on
another machine rather than taken on trust.

Run it with::

    python scripts/bench_training_loop.py

The two loops are kept deliberately identical apart from how a batch of row
indices is produced: same network, same optimiser, same batch size, same
synthetic data at the real problem's shape. They are timed alternately rather
than one after the other, so a laptop that thermally throttles part-way through
penalises both equally instead of whichever ran second.

Synthetic data is used on purpose. The question is how much *Python overhead*
per batch each approach costs, and that depends only on the shapes -- loading
the real QM9 features would add an 82 MB dependency and change nothing.
"""

from __future__ import annotations

import argparse
import statistics
import time

import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from molecular_property_predictor.train import TrainingConfig, shuffled_batches

#: The real problem's shape: the QM9 training split against the sorted Coulomb
#: matrix, at the width the served checkpoint was trained with.
N_ROWS = 104_664
N_FEATURES = 435


def build_model(config: TrainingConfig) -> tuple[nn.Module, torch.optim.Optimizer]:
    """A network of the project's default shape, seeded so both loops match."""
    torch.manual_seed(config.torch_seed)
    layers: list[nn.Module] = []
    width = N_FEATURES
    for hidden in config.hidden_sizes:
        layers += [nn.Linear(width, hidden), nn.ReLU(), nn.Dropout(config.dropout)]
        width = hidden
    layers.append(nn.Linear(width, 1))
    model = nn.Sequential(*layers)
    return model, torch.optim.Adam(model.parameters(), lr=config.learning_rate)


def step(model, optimizer, loss_fn, x_batch, y_batch) -> None:
    """One optimisation step -- identical for both loops, so it cancels out."""
    loss = loss_fn(model(x_batch).squeeze(-1), y_batch)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()


def epoch_with_dataloader(model, optimizer, loss_fn, loader) -> None:
    model.train()
    for x_batch, y_batch in loader:
        step(model, optimizer, loss_fn, x_batch, y_batch)


def epoch_with_shuffled_batches(model, optimizer, loss_fn, x, y, batch_size) -> None:
    model.train()
    for rows in shuffled_batches(len(x), batch_size):
        step(model, optimizer, loss_fn, x[rows], y[rows])


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repeats", type=int, default=5, help="epochs per approach")
    args = parser.parse_args()

    config = TrainingConfig()
    x = torch.randn(N_ROWS, N_FEATURES)
    y = torch.randn(N_ROWS)
    loss_fn = nn.MSELoss()

    dataloader_model, dataloader_optimizer = build_model(config)
    loader = DataLoader(
        TensorDataset(x, y),
        batch_size=config.batch_size,
        shuffle=True,
        num_workers=0,
    )
    manual_model, manual_optimizer = build_model(config)

    # One untimed epoch each: the first pass pays for lazy allocator and kernel
    # setup that no later epoch repeats, and charging that to whichever ran
    # first would be an artefact rather than a measurement.
    epoch_with_dataloader(dataloader_model, dataloader_optimizer, loss_fn, loader)
    epoch_with_shuffled_batches(
        manual_model, manual_optimizer, loss_fn, x, y, config.batch_size
    )

    dataloader_times, manual_times = [], []
    for _ in range(args.repeats):
        start = time.perf_counter()
        epoch_with_dataloader(dataloader_model, dataloader_optimizer, loss_fn, loader)
        dataloader_times.append(time.perf_counter() - start)

        start = time.perf_counter()
        epoch_with_shuffled_batches(
            manual_model, manual_optimizer, loss_fn, x, y, config.batch_size
        )
        manual_times.append(time.perf_counter() - start)

    # Median rather than mean: one scheduling hiccup should not move the answer.
    dataloader_median = statistics.median(dataloader_times)
    manual_median = statistics.median(manual_times)
    saving = (dataloader_median - manual_median) / dataloader_median * 100

    print(f"rows {N_ROWS:,}   features {N_FEATURES}   batch {config.batch_size}   "
          f"threads {torch.get_num_threads()}")
    print(f"  DataLoader          {dataloader_median:6.2f} s/epoch")
    print(f"  shuffled_batches    {manual_median:6.2f} s/epoch")
    print(f"  saving              {saving:6.1f}% of each epoch")


if __name__ == "__main__":
    main()

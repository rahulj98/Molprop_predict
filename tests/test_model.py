"""Tests for the network and the saved artifact.

The artifact tests matter as much as the model ones. Phase 6 has to load a file
written here and reproduce these predictions exactly; a mismatch there would
show up as an API that returns plausible-looking wrong numbers.
"""

from __future__ import annotations

import numpy as np
import pytest
import torch
from sklearn.preprocessing import StandardScaler

from molecular_property_predictor.model import (
    ArtifactMetadata,
    MolecularNet,
    load_artifact,
    predict,
    resolve_device,
    save_artifact,
)


@pytest.fixture
def net() -> MolecularNet:
    torch.manual_seed(0)
    return MolecularNet(n_features=20, hidden_sizes=(16, 8), dropout=0.1)


@pytest.fixture
def scaler() -> StandardScaler:
    generator = np.random.default_rng(0)
    return StandardScaler().fit(generator.normal(size=(50, 20)) * 3.0 + 7.0)


@pytest.fixture
def metadata() -> ArtifactMetadata:
    return ArtifactMetadata(
        representation="sorted_coulomb",
        target="lumo",
        n_features=20,
        hidden_sizes=(16, 8),
        dropout=0.1,
        loss_name="mse",
        split_seed=0,
        torch_seed=0,
        epochs_trained=3,
        best_epoch=2,
        validation_mae_ev=0.5,
    )


# --- The network ------------------------------------------------------------


def test_forward_returns_one_prediction_per_row(net):
    output = net(torch.randn(7, 20))

    assert output.shape == (7,)


def test_output_is_squeezed_to_match_targets(net):
    """A (batch, 1) output against a (batch,) target broadcasts to (batch,
    batch). That trains, converges to something, and is silently wrong."""
    predictions = net(torch.randn(5, 20))
    targets = torch.randn(5)

    assert (predictions - targets).shape == (5,)


def test_network_is_not_merely_linear(net):
    """Without a non-linearity a deep stack collapses to one linear layer.

    Superposition would then hold exactly: f(a) + f(b) == f(a + b) up to the
    bias. It must not.
    """
    net.eval()
    a, b = torch.randn(1, 20) * 3, torch.randn(1, 20) * 3

    with torch.no_grad():
        assert not torch.allclose(net(a) + net(b), net(a + b), atol=1e-3)


def test_dropout_is_active_in_train_mode_and_off_in_eval(net):
    """Scoring in the wrong mode does not raise -- it returns noisier numbers."""
    x = torch.randn(64, 20)

    net.train()
    assert not torch.allclose(net(x), net(x))

    net.eval()
    with torch.no_grad():
        torch.testing.assert_close(net(x), net(x))


def test_backward_reaches_every_parameter(net):
    """A layer left out of the forward pass gets no gradient and never learns."""
    net(torch.randn(8, 20)).sum().backward()

    for name, parameter in net.named_parameters():
        assert parameter.grad is not None, name
        assert torch.any(parameter.grad != 0), name


def test_gradients_accumulate_unless_cleared(net):
    """Documents *why* the training loop calls zero_grad().

    PyTorch adds into .grad rather than overwriting. Two identical backward
    passes therefore give double the gradient -- and nothing warns you.

    eval() first, so dropout does not draw a different mask on each forward
    pass. Without it the two passes are genuinely different computations and
    the doubling would not be exact -- which is its own reminder that dropout
    makes the forward pass stochastic.
    """
    net.eval()
    x = torch.randn(8, 20)

    net(x).sum().backward()
    once = net.layers[0].weight.grad.clone()

    net(x).sum().backward()
    twice = net.layers[0].weight.grad

    torch.testing.assert_close(twice, once * 2)

    net.zero_grad()
    net(x).sum().backward()
    torch.testing.assert_close(net.layers[0].weight.grad, once)


def test_network_can_overfit_a_single_batch():
    """The standard neural-network sanity check.

    A model that cannot drive the loss towards zero on 32 examples it sees
    over and over has a wiring bug -- wrong shapes, a detached graph, a frozen
    layer. Catching that here takes two seconds; catching it after a full
    training run takes an hour and looks like a modelling problem.
    """
    torch.manual_seed(0)
    # dropout=0: with regularisation on, memorising a batch is not possible,
    # which is exactly what dropout is for.
    model = MolecularNet(n_features=10, hidden_sizes=(64, 64), dropout=0.0)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    loss_fn = torch.nn.MSELoss()

    x, y = torch.randn(32, 10), torch.randn(32)
    first_loss = loss_fn(model(x), y).item()

    model.train()
    for _ in range(400):
        loss = loss_fn(model(x), y)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    assert loss.item() < first_loss / 100


# --- Devices ----------------------------------------------------------------


def test_resolve_device_honours_an_explicit_choice():
    assert resolve_device("cpu").type == "cpu"


def test_resolve_device_defaults_to_cuda_only_when_available():
    expected = "cuda" if torch.cuda.is_available() else "cpu"

    assert resolve_device().type == expected


# --- The artifact -----------------------------------------------------------


def test_artifact_round_trip_reproduces_predictions(net, scaler, metadata, tmp_path):
    """The Phase 6 contract: what the API loads must predict what we trained."""
    generator = np.random.default_rng(1)
    raw = generator.normal(size=(11, 20)) * 3.0 + 7.0
    before = predict(net, scaler, raw, device="cpu")

    save_artifact(tmp_path / "model.pt", net, scaler, metadata)
    loaded_net, loaded_scaler, _ = load_artifact(tmp_path / "model.pt")
    after = predict(loaded_net, loaded_scaler, raw, device="cpu")

    np.testing.assert_allclose(before, after, rtol=0, atol=0)


def test_artifact_preserves_the_scaler(net, scaler, metadata, tmp_path):
    """A model without its scaler is not a model -- it is wrong numbers."""
    save_artifact(tmp_path / "model.pt", net, scaler, metadata)
    _, loaded_scaler, _ = load_artifact(tmp_path / "model.pt")

    np.testing.assert_allclose(loaded_scaler.mean_, scaler.mean_)
    np.testing.assert_allclose(loaded_scaler.scale_, scaler.scale_)

    x = np.random.default_rng(2).normal(size=(4, 20))
    np.testing.assert_allclose(loaded_scaler.transform(x), scaler.transform(x))


def test_artifact_preserves_metadata(net, scaler, metadata, tmp_path):
    """A checkpoint whose split seed is unknown cannot be honestly evaluated."""
    save_artifact(tmp_path / "model.pt", net, scaler, metadata)
    _, _, loaded = load_artifact(tmp_path / "model.pt")

    assert loaded == metadata


def test_artifact_loads_onto_cpu(net, scaler, metadata, tmp_path):
    """Guards the deployment path: Phase 8's free tier has no GPU."""
    save_artifact(tmp_path / "model.pt", net, scaler, metadata)
    loaded_net, _, _ = load_artifact(tmp_path / "model.pt", device="cpu")

    assert all(p.device.type == "cpu" for p in loaded_net.parameters())


def test_loaded_model_is_in_eval_mode(net, scaler, metadata, tmp_path):
    """Serving with dropout on would make every request return a different
    answer for the same molecule."""
    save_artifact(tmp_path / "model.pt", net, scaler, metadata)
    loaded_net, _, _ = load_artifact(tmp_path / "model.pt")

    assert not loaded_net.training


# --- Prediction -------------------------------------------------------------


def test_predict_scales_its_input(net, scaler):
    """predict() takes raw features; forgetting to scale is otherwise silent."""
    raw = np.random.default_rng(3).normal(size=(6, 20)) * 3.0 + 7.0

    net.eval()
    with torch.no_grad():
        expected = net(torch.as_tensor(scaler.transform(raw), dtype=torch.float32))

    np.testing.assert_allclose(
        predict(net, scaler, raw, device="cpu"), expected.numpy(), rtol=1e-6
    )


def test_predict_is_unaffected_by_batching(net, scaler):
    """Batching is a memory strategy, not part of the calculation."""
    raw = np.random.default_rng(4).normal(size=(37, 20))

    np.testing.assert_allclose(
        predict(net, scaler, raw, device="cpu", batch_size=4),
        predict(net, scaler, raw, device="cpu", batch_size=1024),
        rtol=1e-6,
    )


def test_predict_returns_one_value_per_molecule(net, scaler):
    raw = np.random.default_rng(5).normal(size=(13, 20))

    assert predict(net, scaler, raw, device="cpu").shape == (13,)

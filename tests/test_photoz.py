import copy

import numpy as np
import pandas as pd
import pytest
import torch
import torch.distributions as distributions
from torch import nn
from torch.optim import Adam
from torch.utils.data import DataLoader, TensorDataset

from kids.ml.photoz.regressors import (
    BayesianNet,
    BNConfig,
    MixtureDensityNetwork,
    MDNConfig,
    MixtureNormalLayer,
)
from kids.ml.photoz.train_bnn import (
    BNNTrainingConfig,
    bnn_train_one_epoch,
    bnn_validate,
    train_bnn,
)
from kids.ml.photoz.train_core import EarlyStopping, get_data_loader
from kids.ml.photoz.train_mdn import mdn_train_one_epoch, mdn_validate


class SchedulerSpy:
    def __init__(self):
        self.calls = []

    def step(self, *args):
        self.calls.append(args)


class SimpleMDN(nn.Module):
    def __init__(self):
        super().__init__()
        self.location = nn.Linear(2, 1)

    def forward(self, x):
        loc = self.location(x).squeeze(-1)
        return distributions.Normal(loc, torch.ones_like(loc))


class SimpleBNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.location = nn.Linear(2, 1)

    def forward(self, x):
        loc = self.location(x).squeeze(-1)
        return distributions.Normal(loc, torch.ones_like(loc)), self.location.weight.square().sum()


@pytest.fixture
def small_data():
    features = np.arange(12, dtype=np.float32).reshape(6, 2)
    target = pd.Series(np.arange(6, dtype=np.float32))
    return features, target


@pytest.fixture
def small_loader():
    x = torch.tensor([[0.0, 0.0], [1.0, 1.0], [2.0, 2.0], [3.0, 3.0]])
    y = torch.tensor([0.0, 1.0, 2.0, 3.0])
    return DataLoader(TensorDataset(x, y), batch_size=2, shuffle=False)


def test_get_data_loader_preserves_rows_and_dtypes(small_data):
    features, target = small_data

    loader = get_data_loader(features, target, batch_size=4, num_workers=0)

    batches = list(loader)
    assert len(loader.dataset) == 6
    assert [batch[0].shape[0] for batch in batches] == [4, 2]
    assert batches[0][0].dtype == torch.float32
    assert batches[0][1].dtype == torch.float32
    torch.testing.assert_close(batches[0][0], torch.tensor(features[:4]))


def test_mixture_normal_layer_returns_finite_distribution():
    layer = MixtureNormalLayer(n_mixture_components=2)
    output = layer(torch.zeros(3, 6))

    assert output.batch_shape == (3,)
    assert output.event_shape == torch.Size()
    assert torch.all(output.component_distribution.scale > 0)
    assert torch.isfinite(output.log_prob(torch.zeros(3))).all()


def test_mdn_forward_returns_expected_distribution_shape():
    model = MixtureDensityNetwork(
        MDNConfig(n_features=2, n_neurons=4, n_mixture_components=2)
    )
    output = model(torch.randn(3, 2))

    assert output.batch_shape == (3,)
    assert torch.isfinite(output.log_prob(torch.zeros(3))).all()


def test_bnn_forward_returns_distribution_and_scalar_kl():
    model = BayesianNet(BNConfig(n_features=2, n_neurons=4, n_mixture_components=2))
    output, kl_loss = model(torch.randn(3, 2))

    assert output.batch_shape == (3,)
    assert kl_loss.ndim == 0
    assert torch.isfinite(output.log_prob(torch.zeros(3))).all()
    assert torch.isfinite(kl_loss)


def test_mdn_training_steps_scheduler_once_per_batch(small_loader):
    model = SimpleMDN()
    optimizer = Adam(model.parameters(), lr=0.01)
    scheduler = SchedulerSpy()

    loss = mdn_train_one_epoch(
        model, small_loader, optimizer, device="cpu", scheduler=scheduler
    )

    assert np.isfinite(loss)
    assert len(scheduler.calls) == len(small_loader)
    assert all(call == () for call in scheduler.calls)


def test_mdn_validation_returns_mean_batch_loss(small_loader):
    model = SimpleMDN()

    expected = []
    with torch.no_grad():
        for x, y in small_loader:
            expected.append((-model(x).log_prob(y).mean()).item())

    actual = mdn_validate(model, small_loader, device="cpu")

    assert actual == pytest.approx(sum(expected) / len(expected))


def test_bnn_training_includes_kl_in_optimization_loss(small_loader):
    model = SimpleBNN()
    initial_parameters = copy.deepcopy(model.state_dict())
    optimizer = Adam(model.parameters(), lr=0.01)

    loss = bnn_train_one_epoch(model, small_loader, optimizer, device="cpu")

    assert np.isfinite(loss)
    assert any(
        not torch.equal(initial_parameters[name], parameter)
        for name, parameter in model.state_dict().items()
    )


def test_bnn_validation_excludes_kl_term(small_loader):
    model = SimpleBNN()

    expected = []
    with torch.no_grad():
        for x, y in small_loader:
            distribution, _ = model(x)
            expected.append((-distribution.log_prob(y).mean()).item())

    actual = bnn_validate(model, small_loader, device="cpu")

    assert actual == pytest.approx(sum(expected) / len(expected))


def test_bnn_training_steps_plateau_scheduler_once_per_epoch(small_loader):
    model = SimpleBNN()
    optimizer = Adam(model.parameters(), lr=0.01)
    scheduler = SchedulerSpy()
    config = BNNTrainingConfig(
        device="cpu",
        n_epochs=3,
        early_stopping_patience=10,
    )

    history = train_bnn(model, small_loader, small_loader, optimizer, scheduler, config)

    assert len(history) == 3
    assert len(scheduler.calls) == 3
    assert all(len(call) == 1 for call in scheduler.calls)
    assert all(torch.isfinite(torch.tensor(call[0])) for call in scheduler.calls)


def test_early_stopping_keeps_best_state():
    model = nn.Linear(1, 1)
    stopping = EarlyStopping(patience=2, min_delta=0.0)

    model.weight.data.fill_(1.0)
    assert stopping.update(1.0, model) is False
    best_state = copy.deepcopy(model.state_dict())

    model.weight.data.fill_(2.0)
    assert stopping.update(2.0, model) is False
    assert stopping.update(3.0, model) is True

    assert torch.equal(stopping.best_state["weight"], best_state["weight"])

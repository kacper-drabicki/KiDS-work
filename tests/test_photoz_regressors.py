import torch

from kids.photoz_regressors import (
    BNConfig,
    BayesianNet,
    MDNConfig,
    MixtureDensityNetwork,
    MixtureNormalLayer,
)


def test_mixture_layer_returns_positive_scales():
    layer = MixtureNormalLayer(n_mixture_components=2)
    raw_output = torch.zeros((4, 6))
    distribution = layer(raw_output)
    assert distribution.batch_shape == torch.Size([4])
    assert (distribution.component_distribution.scale > 0).all()
    torch.testing.assert_close(distribution.mixture_distribution.probs.sum(dim=-1), torch.ones(4))


def test_mdn_returns_distribution_for_configured_feature_count():
    model = MixtureDensityNetwork(MDNConfig(n_mixture_components=2, n_features=3, n_neurons=4))
    inputs = torch.randn(4, 3)
    distribution = model(inputs)
    assert distribution.batch_shape == torch.Size([4])
    assert torch.isfinite(distribution.log_prob(torch.zeros(4))).all()


def test_bayesian_net_returns_distribution_and_finite_kl():
    model = BayesianNet(BNConfig(n_mixture_components=2, n_features=3, n_neurons=4))
    inputs = torch.randn(4, 3)
    distribution, kl = model(inputs)
    assert distribution.batch_shape == torch.Size([4])
    assert kl.ndim == 0
    assert torch.isfinite(kl)
    assert kl.item() >= 0.0

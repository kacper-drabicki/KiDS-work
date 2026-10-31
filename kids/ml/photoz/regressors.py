
import torch
import torch.nn as nn
import torch.distributions as D
import torch.nn.functional as F
from bayesian_torch.layers import LinearFlipout
from dataclasses import dataclass

# ------------------------------------------ 

class MixtureNormalLayer(nn.Module):
    def __init__(self, n_mixture_components: int):
        super().__init__()
        self.n_mixture_components = n_mixture_components

    def forward(self, x: torch.Tensor) -> D.Distribution:
        # Split the input tensor into logits, locs (means), and scales (stds)
        pi_logits, locs, raw_scales = torch.split(
            x, self.n_mixture_components, dim=-1)
        
        # Standard deviations must be positive; add epsilon for numerical stability
        scales = F.softplus(raw_scales + 0.5413) + 1e-5 # 0.5413 to match TF implementation
        
        # Construct the mixture distribution
        categorical = D.Categorical(logits=pi_logits)
        component = D.Normal(loc=locs, scale=scales)
        
        return D.MixtureSameFamily(categorical, component)

# ------------------------------------------ 

@dataclass
class MDNConfig:
    n_mixture_components: int = 5
    n_features: int = 55
    n_neurons: int = 512
    dropout_rate: float = 0.2

# ------------------------------------------

class MixtureDensityNetwork(nn.Module):
    def __init__(self, config: MDNConfig):
        super().__init__()
        self.config = config

        self._build_model()
        self.apply(self._init_weights)

    def _build_model(self):
        layers = [nn.Linear(self.config.n_features, self.config.n_neurons),
                  nn.ReLU(),
                  nn.Linear(self.config.n_neurons, self.config.n_neurons),
                  nn.ReLU(),
                  nn.Dropout(self.config.dropout_rate)]
        
        for _ in range(4):
            layers.extend([nn.Linear(self.config.n_neurons, self.config.n_neurons),
                           nn.ReLU(),
                           nn.Linear(self.config.n_neurons, self.config.n_neurons),
                           nn.ReLU(),
                           nn.Dropout(self.config.dropout_rate)])

        layers.extend([
            nn.Linear(self.config.n_neurons, self.config.n_neurons),
            nn.ReLU(),
            nn.Linear(self.config.n_neurons, self.config.n_neurons),
            nn.ReLU(),
            nn.Linear(self.config.n_neurons, 3 * self.config.n_mixture_components) # K mixing weights (logits), K means (locs), K standard deviations (scales)
        ])

        self.backbone = nn.Sequential(*layers)
        self.mixture_layer = MixtureNormalLayer(self.config.n_mixture_components)

    @staticmethod
    def _init_weights(m: nn.Module):
        if isinstance(m, nn.Linear):
            nn.init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='relu')
            if m.bias is not None:
                nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor) -> D.Distribution:
        raw_output = self.backbone(x)
        return self.mixture_layer(raw_output)

    def save(self, path: str) -> None:
        torch.save(self.state_dict(), path)

    def load_weights(self, path: str, device: str) -> None:
        self.load_state_dict(torch.load(path, map_location=torch.device(device)))
        
        
# ------------------------------------------ 

@dataclass
class BNConfig:
    n_mixture_components: int = 5
    n_features: int = 55
    n_neurons: int = 512

# ------------------------------------------ 

class BayesianNet(nn.Module):
    def __init__(self, config: BNConfig):
        super().__init__()
        self.config = config

        self._build_model()

    def _build_model(self):
        layers = [nn.Linear(self.config.n_features, self.config.n_neurons),
                  nn.ReLU(),
                  nn.BatchNorm1d(self.config.n_neurons)]
        
        for _ in range(8):
            layers.extend([nn.Linear(self.config.n_neurons, self.config.n_neurons),
                           nn.ReLU(),
                           nn.BatchNorm1d(self.config.n_neurons)])
        
        self.backbone = nn.Sequential(*layers)

        self.blinear1 = LinearFlipout(self.config.n_neurons, self.config.n_neurons)
        self.blinear2 = LinearFlipout(self.config.n_neurons, self.config.n_neurons)
        self.blinear3 = LinearFlipout(self.config.n_neurons, self.config.n_neurons)
        self.b_activation = nn.Tanh()

        self.output_layer = nn.Linear(self.config.n_neurons, 3 * self.config.n_mixture_components) # K mixing weights (logits), K means (locs), K standard deviations (scales)

        self.mixture_layer = MixtureNormalLayer(self.config.n_mixture_components)

    def _init_weights(self, m: nn.Module):
        pass

    def forward(self, x: torch.Tensor) -> tuple[D.Distribution, torch.Tensor]:
        
        out = self.backbone(x)
        out, kl1 = self.blinear1(out)
        out = self.b_activation(out)
        out, kl2 = self.blinear2(out)
        out = self.b_activation(out)
        out, kl3 = self.blinear3(out)
        out = self.b_activation(out)
        out = self.output_layer(out)

        total_kl = kl1 + kl2 + kl3

        return self.mixture_layer(out), total_kl

    def save(self, path: str) -> None:
        torch.save(self.state_dict(), path)

    def load_weights(self, path: str, device: str) -> None:
        self.load_state_dict(torch.load(path, map_location=torch.device(device)))


import torch
import torch.nn as nn
import torch.distributions as D
import torch.nn.functional as F
from dataclasses import dataclass

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
            nn.ReLU()
        ])

        self.backbone = nn.Sequential(*layers)
        # Output parameter projection:
        # K mixing weights (logits), K means (locs), K standard deviations (scales)
        self.param_layer = nn.Linear(self.config.n_neurons, 3 * self.config.n_mixture_components)

    @staticmethod
    def _init_weights(m: nn.Module):
        if isinstance(m, nn.Linear):
            nn.init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='relu')
            if m.bias is not None:
                nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor) -> D.Distribution:
        features = self.backbone(x)
        raw_params = self.param_layer(features)
        
        # Split outputs into logits, locs (means), and scales (stds)
        pi_logits, locs, raw_scales = torch.split(
            raw_params, self.config.n_mixture_components, dim=-1
        )
        
        # Standard deviations must be positive; add epsilon for numerical stability
        scales = F.softplus(raw_scales + 0.5413) + 1e-5 # 0.5413 to match TF implementation
        
        # Construct the mixture distribution
        categorical = D.Categorical(logits=pi_logits)
        component = D.Normal(loc=locs, scale=scales)
        
        return D.MixtureSameFamily(categorical, component)
        
# ------------------------------------------ 


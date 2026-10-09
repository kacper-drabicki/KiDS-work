
import torch
import torch.nn as nn
import torch.distributions as D
import torch.nn.functional as F
from numpy import nan
from dataclasses import dataclass,  asdict
from xgboost import XGBClassifier

@dataclass
class XGBConfig:
    booster: str = 'gbtree'
    objective: str = 'multi:softmax'
    n_estimators: int = 100000
    early_stopping_rounds: int = 20
    max_depth: int = 7
    learning_rate: float = 0.1
    gamma: float = 0.0
    min_child_weight: float = 5.0
    max_delta_step: float = 0.0
    colsample_bytree: float = 0.9
    colsample_bylevel: float = 1.0
    subsample: float = 0.8
    reg_alpha: float = 0.1
    reg_lambda: float = 1.0
    base_score: float = 0.5
    random_state: int = 18235
    missing: float = nan
    verbosity: int = 0
    n_jobs: int = 32

@dataclass
class MDNConfig:
    n_mixture_components: int = 5
    n_features: int = 55
    n_neurons: int = 512
    dropout_rate: float = 0.2

# ------------------------------------------

class XGBClf:
    def __init__(self, config=None):
        self.model = XGBClassifier(**asdict(config)) if config is not None else XGBClassifier()

    def train(self, X_train, y_train, X_val, y_val, verbose: bool = True):
        self.model.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=verbose)

    def predict(self, X):
        return self.model.predict(X)

    def save(self, filepath: str):
        # XGBoost automatically infers format from extension (.json or .ubj)
        self.model.save_model(filepath)

    def load(self, filepath: str):
        self.model.load_model(filepath)

# ------------------------------------------

class MixtureDensityNetwork(nn.Module):
    def __init__(self, config):
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


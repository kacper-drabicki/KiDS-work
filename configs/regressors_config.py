from dataclasses import dataclass

@dataclass(frozen=True)
class MDNConfig:
    n_mixture_components: int = 5
    n_features: int = 55
    n_neurons: int = 512
    dropout_rate: float = 0.2

# ------------------------------------------

@dataclass(frozen=True)
class BNConfig:
    n_mixture_components: int = 3
    n_features: int = 55
    n_neurons: int = 512

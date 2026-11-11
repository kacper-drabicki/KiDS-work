from dataclasses import dataclass

@dataclass(frozen=True)
class SOMConfig:
    x: int = 60
    y: int = 60
    num_features: int = 91
    epochs: int = 35
    batch_size: int = 4 * 1024
    learning_rate: float = 0.8
    sigma: float = 30
    topology: str = "rectangular"
    pbc: bool = True
    lr_decay_function: str = "asymptotic_decay"
    sigma_decay_function: str = "lr_linear_decay_to_zero"
    neighborhood_function: str = "gaussian"
    distance_function: str = "euclidean"
    initialization_mode: str = "pca"
    random_seed: int = 6487362742

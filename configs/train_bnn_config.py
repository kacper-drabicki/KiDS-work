import torch
from dataclasses import dataclass

@dataclass
class BNNTrainingConfig:
    device: str = 'cuda' if torch.cuda.is_available() else 'cpu'
    batch_size: int = 256 # unfortunately, larger batch size doesn't generalize well
    learning_rate: float = 1e-3
    n_epochs: int = 400
    # Early stopping parameters
    early_stopping_patience: int = 50
    min_delta: float = 0.0
    # Scheduler parameters
    mode: str = 'min'
    factor: float = 0.1
    scheduler_patience: int = 10

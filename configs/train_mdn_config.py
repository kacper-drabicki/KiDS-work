import torch
from dataclasses import dataclass

@dataclass
class MDNTrainingConfig:
    device: str = 'cuda' if torch.cuda.is_available() else 'cpu'
    batch_size: int = 2048
    learning_rate: float = 3e-4
    n_epochs: int = 200
    # Early stopping parameters
    early_stopping_patience: int = 30
    min_delta: float = 0.0
    # Scheduler parameters
    max_lr: float = 3e-4

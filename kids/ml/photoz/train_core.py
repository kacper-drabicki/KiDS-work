import copy
import pandas as pd
import numpy as np
import torch
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader, TensorDataset
from dataclasses import dataclass

def get_data_loader(X: np.ndarray, y: pd.Series, batch_size: int, num_workers: int = 16) -> DataLoader:
    X_tensor = torch.tensor(X, dtype=torch.float32)
    y_tensor = torch.tensor(y.values, dtype=torch.float32)

    return DataLoader(TensorDataset(X_tensor, y_tensor),
                      batch_size=batch_size,
                      pin_memory=True,
                      num_workers=num_workers,
                      persistent_workers=num_workers > 0)

@dataclass
class EarlyStopping:
    patience: int = 10
    min_delta: float = 0.0

    best_loss: float = float("inf")
    epochs_without_improvement: int = 0
    best_state: dict | None = None

    def update(self, val_loss: float, model) -> bool:
        improved = val_loss < self.best_loss - self.min_delta

        if improved:
            self.best_loss = val_loss
            self.epochs_without_improvement = 0
            self.best_state = copy.deepcopy(model.state_dict())
        else:
            self.epochs_without_improvement += 1

        return self.epochs_without_improvement >= self.patience

def plot_training_history(history: list[dict[str, float]]) -> None:

    train_losses = [h["train_loss"] for h in history]
    val_losses = [h["val_loss"] for h in history]

    plt.figure(figsize=(10, 6))
    plt.plot(train_losses, label="Train Loss")
    plt.plot(val_losses, label="Validation Loss")
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.title("Training and Validation Loss Over Epochs")
    plt.legend()
    plt.grid()
    plt.show()

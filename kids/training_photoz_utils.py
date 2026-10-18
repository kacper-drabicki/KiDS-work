import copy
import pandas as pd
import numpy as np
import torch
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

def train_one_epoch(model, train_loader: DataLoader, optimizer, device: str, scheduler = None) -> float:
    model.train()   
    train_loss = 0.0    

    for x, y in train_loader:
        x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)

        distribution = model(x)
        batch_loss = -distribution.log_prob(y).mean()
        batch_loss.backward()
        optimizer.step()

        if scheduler is not None:
            scheduler.step()

        train_loss += batch_loss.item()

    return train_loss / len(train_loader)

@torch.no_grad()
def validate(model, val_loader: DataLoader, device: str) -> float:
    model.eval()
    val_loss = 0.0

    for x, y in val_loader:
        x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)

        distribution = model(x)
        batch_loss = -distribution.log_prob(y).mean()
        val_loss += batch_loss.item()

    return val_loss / len(val_loader)

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

def train_model(model, train_loader: DataLoader, val_loader: DataLoader,
                optimizer, device: str, n_epochs: int, patience: int = 10,
                min_delta: float = 0, scheduler = None) -> list[dict[str, float]]:
    early_stopping = EarlyStopping(patience=patience, min_delta=min_delta)

    history = []
    
    for epoch in range(n_epochs):
        train_loss = train_one_epoch(model, train_loader, optimizer, device, scheduler)
        val_loss = validate(model, val_loader, device)

        history.append({"train_loss": train_loss, "val_loss": val_loss})

        print(f"Epoch {epoch + 1}/{n_epochs} - Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}")

        if early_stopping.update(val_loss, model):
            print(f"Early stopping at epoch {epoch + 1}")
            break

    if early_stopping.best_state is not None:
        model.load_state_dict(early_stopping.best_state)

    return history

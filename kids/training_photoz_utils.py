import pandas as pd
import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset

def get_data_loader(X: np.ndarray, y: pd.Series, batch_size: int, num_workers: int = 16) -> DataLoader:
    X_tensor = torch.tensor(X, dtype=torch.float32)
    y_tensor = torch.tensor(y.values, dtype=torch.float32)

    return DataLoader(TensorDataset(X_tensor, y_tensor),
                      batch_size=batch_size,
                      pin_memory=True,
                      num_workers=num_workers,
                      persistent_workers=num_workers > 0)

def train_one_epoch(model, train_loader: DataLoader, optimizer, device: str) -> float:
    model.train()   
    train_loss = 0.0    

    for x, y in train_loader:
        x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)

        distribution = model(x)
        batch_loss = -distribution.log_prob(y).sum()
        batch_loss.backward()
        optimizer.step()

        train_loss += batch_loss.item()

    return train_loss / len(train_loader.dataset)

@torch.no_grad()
def validate(model, val_loader: DataLoader, device: str) -> float:
    model.eval()
    val_loss = 0.0

    for x, y in val_loader:
        x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)

        distribution = model(x)
        batch_loss = -distribution.log_prob(y).sum()
        val_loss += batch_loss.item()

    return val_loss / len(val_loader.dataset)

def train_model(model, train_loader: DataLoader, val_loader: DataLoader, optimizer, device: str, n_epochs: int) -> None:
    for epoch in range(n_epochs):
        train_loss = train_one_epoch(model, train_loader, optimizer, device)
        val_loss = validate(model, val_loader, device)

        print(f"Epoch {epoch + 1}/{n_epochs} - Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}")

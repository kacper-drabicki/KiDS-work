import torch
from dataclasses import dataclass
from torch.utils.data import DataLoader
from kids.logger import logger
from kids.ml.photoz.train_core import EarlyStopping


def bnn_train_one_epoch(model, train_loader: DataLoader, optimizer, device: str) -> float:
    model.train()   
    train_loss = 0.0    

    for x, y in train_loader:
        x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)

        distribution, kl_loss = model(x)
        batch_loss = -distribution.log_prob(y).mean()
        total_loss = batch_loss + kl_loss / len(train_loader.dataset)
        total_loss.backward()
        optimizer.step()

        train_loss += total_loss.item()

    return train_loss / len(train_loader)


@torch.no_grad()
def bnn_validate(model, val_loader: DataLoader, device: str) -> float:
    model.eval()
    val_loss = 0.0

    for x, y in val_loader:
        x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)

        distribution, kl_loss = model(x)
        batch_loss = -distribution.log_prob(y).mean() 
        val_loss += batch_loss.item() # We intentionally do not include KL loss in validation loss

    return val_loss / len(val_loader)

@dataclass
class BNNTrainingConfig:
    device: str = 'cuda' if torch.cuda.is_available() else 'cpu'
    batch_size: int = 2048
    learning_rate: float = 3e-4
    n_epochs: int = 200
    # Early stopping parameters
    early_stopping_patience: int = 30
    min_delta: float = 0.0
    # Scheduler parameters
    mode: str = 'min'
    factor: float = 0.5
    scheduler_patience: int = 10

def train_bnn(model, train_loader: DataLoader, val_loader: DataLoader,
                optimizer, scheduler, config: BNNTrainingConfig) -> list[dict[str, float]]:
    
    early_stopping = EarlyStopping(patience=config.early_stopping_patience, min_delta=config.min_delta)

    history = []
    
    for epoch in range(config.n_epochs):
        train_loss = bnn_train_one_epoch(model, train_loader, optimizer, config.device)
        val_loss = bnn_validate(model, val_loader, config.device)

        if scheduler is not None:
            scheduler.step(val_loss)
        
        history.append({"train_loss": train_loss, "val_loss": val_loss})

        logger.info(f"Epoch {epoch + 1}/{config.n_epochs} - Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}")

        if early_stopping.update(val_loss, model):
            logger.info(f"Early stopping at epoch {epoch + 1}")
            break

    if early_stopping.best_state is not None:
        model.load_state_dict(early_stopping.best_state)

    return history

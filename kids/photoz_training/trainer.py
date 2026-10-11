import torch

class Trainer:
    def __init__(self, model, optimizer, loss_fn, device):
        self.model = model
        self.optimizer = optimizer
        self.loss_fn = loss_fn
        self.device = device

    def _train_epoch(self, train_loader):
        self.model.train()
        total_loss = 0
        
        for x, y in train_loader: 
            x = x.to(self.device, non_blocking=True)
            y = y.to(self.device, non_blocking=True)

            self.optimizer.zero_grad(set_to_none=True)
            loss = self.loss_fn(self.model(x), y)
            loss.backward()
            self.optimizer.step()

            total_loss += loss.item()

        return total_loss / len(train_loader.dataset)

    @torch.no_grad()
    def _validate(self, val_loader):
        self.model.eval()
        total_loss = 0.0

        for x, y in val_loader:
            x = x.to(self.device, non_blocking=True)
            y = y.to(self.device, non_blocking=True)
            loss = self.loss_fn(self.model(x), y)
            total_loss += loss.item()

        return total_loss / len(val_loader.dataset)

    def train(self, train_loader, val_loader, n_epochs):
        history = {"train": [], "val": []}

        for epoch in range(n_epochs):
            train_loss = self._train_epoch(train_loader)
            val_loss = self._validate(val_loader)

            history["train"].append(train_loss)
            history["val"].append(val_loss)

            print(
                f"Epoch {epoch + 1}/{n_epochs} "
                f"- train: {train_loss:.5f} "
                f"- val: {val_loss:.5f}"
            )

        return history


    

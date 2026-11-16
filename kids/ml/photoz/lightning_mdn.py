import lightning as L
import torch
from torch.distributions import Distribution
from torch.optim import Adam
from torch.optim.lr_scheduler import OneCycleLR

from configs.regressors_config import MDNConfig
from configs.train_mdn_config import MDNTrainingConfig
from kids.ml.photoz.regressors import MixtureDensityNetwork


class MDNLightningModule(L.LightningModule):
    def __init__(
        self,
        model_config: MDNConfig,
        training_config: MDNTrainingConfig,
    ) -> None:
        super().__init__()
        self.save_hyperparameters(ignore=["model_config", "training_config"])
        self.model = MixtureDensityNetwork(model_config)
        self.learning_rate = training_config.learning_rate
        self.max_lr = training_config.max_lr

    def forward(self, features: torch.Tensor) -> Distribution:
        return self.model(features)

    def _loss(self, batch: tuple[torch.Tensor, torch.Tensor]) -> torch.Tensor:
        features, target = batch
        return -self(features).log_prob(target).mean()

    def training_step(
        self,
        batch: tuple[torch.Tensor, torch.Tensor],
        batch_idx: int,
    ) -> torch.Tensor:
        loss = self._loss(batch)
        self.log("train_loss", loss, on_epoch=True, prog_bar=True)
        return loss

    def validation_step(
        self,
        batch: tuple[torch.Tensor, torch.Tensor],
        batch_idx: int,
    ) -> None:
        self.log("val_loss", self._loss(batch), on_epoch=True, prog_bar=True)

    def configure_optimizers(self) -> dict:
        optimizer = Adam(self.parameters(), lr=self.learning_rate)
        scheduler = OneCycleLR(
            optimizer,
            max_lr=self.max_lr,
            total_steps=self.trainer.estimated_stepping_batches,
        )
        return {
            "optimizer": optimizer,
            "lr_scheduler": {"scheduler": scheduler, "interval": "step"},
        }

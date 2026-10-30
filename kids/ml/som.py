from dataclasses import dataclass

import torch
from torch.utils.data import DataLoader, TensorDataset
from torchsom.core import SOM
from tqdm.auto import tqdm


@dataclass(frozen=True)
class SOMConfig:
    preprocessor_sample_size: int = 2_500_000
    random_state: int = 42
    x: int = 60
    y: int = 60
    epochs: int = 35
    batch_size: int = 4 * 1024
    learning_rate: float = 0.8
    sigma: float = 30
    device: str = "auto"
    topology: str = "rectangular"
    pbc: bool = True
    lr_decay_function: str = "asymptotic_decay"
    sigma_decay_function: str = "lr_linear_decay_to_zero"
    neighborhood_function: str = "gaussian"
    distance_function: str = "euclidean"
    initialization_mode: str = "pca"

    def resolved_device(self) -> str:
        if self.device == "auto":
            return "cuda" if torch.cuda.is_available() else "cpu"
        if self.device == "cuda" and not torch.cuda.is_available():
            raise RuntimeError("SOM device 'cuda' was requested, but CUDA is unavailable")
        return self.device

    def model_parameters(self, num_features: int) -> dict[str, object]:
        return {
            "x": self.x,
            "y": self.y,
            "num_features": num_features,
            "epochs": self.epochs,
            "batch_size": self.batch_size,
            "learning_rate": self.learning_rate,
            "sigma": self.sigma,
            "device": self.resolved_device(),
            "topology": self.topology,
            "pbc": self.pbc,
            "lr_decay_function": self.lr_decay_function,
            "sigma_decay_function": self.sigma_decay_function,
            "neighborhood_function": self.neighborhood_function,
            "distance_function": self.distance_function,
            "initialization_mode": self.initialization_mode,
        }


class MySOM(SOM):
    def fit(self, data: torch.Tensor, verbose: bool = True) -> tuple[list[float], list[float]]:
        ''' Modfied dataloader to use pin_memory and num_workers for faster training on GPU. '''
        dataset = TensorDataset(data)
        dataloader = DataLoader(
            dataset,
            batch_size=self.batch_size,
            shuffle=True,
            pin_memory=True,
            num_workers=8)

# ---------- UNMODIFIED CODE ----------

        epoch_q_errors = []
        epoch_t_errors = []
        for epoch in tqdm(
            range(self.epochs),
            desc="Training SOM",
            unit="epoch",
            disable=not verbose,
        ):
            # Update learning parameters thanks to decay functions (schedulers)
            lr = self.lr_schedule[epoch]
            sigma = self.sigma_schedule[epoch]
            batch_q_errors = []
            batch_t_errors = []
            for batch in dataloader:
                batch_data = batch[0].to(self.device)
                # Rerieve the BMUs [batch_size, 2]
                bmus = self.identify_bmus(batch_data)
                # Update the weights
                self._update_weights(batch_data, bmus, lr, sigma)
                # Calculate batch errors
                batch_q_errors.append(self.quantization_error(batch_data))
                batch_t_errors.append(self.topographic_error(batch_data))
            # Calculate average epoch errors
            epoch_q_errors.append(torch.tensor(batch_q_errors).mean().item())
            epoch_t_errors.append(100 * torch.tensor(batch_t_errors).mean().item())

        return epoch_q_errors, epoch_t_errors

    _VALID_RETRIEVAL_MODES = ("bmu_only", "bmu_neighborhood", "bmu_neighborhood_knn")

# -------------------------------------

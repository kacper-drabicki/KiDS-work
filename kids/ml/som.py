import torch
from torch.utils.data import DataLoader, TensorDataset
from torchsom.core import SOM
from tqdm.auto import tqdm

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

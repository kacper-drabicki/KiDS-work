import pandas as pd
import torch
from pathlib import Path
from kids.logger import logger
from kids.ml.classification.xgbclf import XGBClf
from kids.catalog.cleaning import clean_mask, replace_sentinel
from kids.catalog.features import add_colors, add_ratios
from kids.ml.training_data import TrainingData

class Inferencer:
    def __init__(self, input_path: Path, preprocessor, features: list[str], index_col = None, chunksize: int = 1_000_000) -> None:
        self.input_path = input_path
        self.preprocessor = preprocessor
        self.chunksize = chunksize
        self.index_col = index_col
        self.features = features

    @staticmethod
    def _prepare_chunk(chunk: pd.DataFrame) -> pd.DataFrame:

        chunk = clean_mask(chunk)
        chunk = replace_sentinel(chunk)
        chunk = add_colors(chunk)
        chunk = add_ratios(chunk)

        # return chunk.pipe(clean_mask).pipe(replace_sentinel).pipe(add_colors)
        return chunk

    def infer_class(self, model: XGBClf, output_path: Path) -> None:
        id_to_class = TrainingData.ID_TO_CLASS
        
        wrote_header = False

        total_rows_written = 0

        for chunk in pd.read_csv(self.input_path, chunksize=self.chunksize, index_col=self.index_col):
            prepared_chunk = self._prepare_chunk(chunk)
            
            if prepared_chunk.empty:
                continue
                
            features = prepared_chunk.loc[:, self.features]
            transformed_features = self.preprocessor.transform(features)

            prediction_ids = model.predict(transformed_features).astype(int)

            output_chunk = chunk.loc[prepared_chunk.index].copy()
            output_chunk['PREDICTED_CLASS'] = pd.Series(prediction_ids, index=prepared_chunk.index).map(id_to_class).fillna("Unknown")
            
            output_chunk.to_csv(
                output_path, 
                mode="w" if not wrote_header else "a", 
                header=not wrote_header, 
                index=False
            )
            
            chunk_len = len(output_chunk)
            total_rows_written += chunk_len
            logger.info(f"Wrote {chunk_len} rows to {output_path}. Total rows written: {total_rows_written}")
            wrote_header = True

    @staticmethod
    def _get_distribution(model_output):
        if isinstance(model_output, tuple):
            return model_output[0]
        return model_output

    def infer_quasar_photoz(
        self,
        model: torch.nn.Module,
        output_path: Path,
        device: str = "cpu",
        n_samples: int = 1,
    ) -> None:
        if n_samples < 1:
            raise ValueError("n_samples must be at least 1")

        model = model.to(device)
        model.eval()
        wrote_header = False
        total_rows_written = 0

        for chunk in pd.read_csv(
            self.input_path, chunksize=self.chunksize, index_col=self.index_col
        ):
            prepared_chunk = self._prepare_chunk(chunk)

            if prepared_chunk.empty:
                continue

            features = prepared_chunk.loc[:, self.features]
            transformed_features = self.preprocessor.transform(features)
            input_tensor = torch.as_tensor(
                transformed_features.copy(), dtype=torch.float32, device=device
            )

            sample_means = []
            sample_variances = []
            with torch.no_grad():
                for _ in range(n_samples):
                    distribution = self._get_distribution(model(input_tensor))
                    sample_means.append(distribution.mean)
                    sample_variances.append(distribution.variance)

            means = torch.stack(sample_means)
            variances = torch.stack(sample_variances)
            predicted_z = means.mean(dim=0)
            predicted_z_std = torch.sqrt(variances.mean(dim=0) + means.var(dim=0, unbiased=False))

            if not torch.isfinite(predicted_z).all() or not torch.isfinite(predicted_z_std).all():
                raise ValueError("Photo-z model produced non-finite predictions")

            output_chunk = chunk.loc[prepared_chunk.index].copy()
            output_chunk["PREDICTED_Z"] = predicted_z.cpu().numpy()
            output_chunk["PREDICTED_Z_STD"] = predicted_z_std.cpu().numpy()
            output_chunk.to_csv(
                output_path,
                mode="w" if not wrote_header else "a",
                header=not wrote_header,
                index=False,
            )

            chunk_len = len(output_chunk)
            total_rows_written += chunk_len
            logger.info(
                f"Wrote {chunk_len} rows to {output_path}. "
                f"Total rows written: {total_rows_written}"
            )
            wrote_header = True

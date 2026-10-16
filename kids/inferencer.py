# import logging / for laters
import pandas as pd
from pathlib import Path
from kids.classifiers import XGBClf
from kids.catalog.catalog import clean_mask, replace_sentinel, add_colors, add_ratios
from kids.features import COLORS, MAGNITUDES
from kids.training_data import TrainingData

class Inferencer:
    def __init__(self, input_path: Path, preprocessor, features: list[str], index_col: int = None, chunksize: int = 10_000_000) -> None:
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
            output_chunk['PREDICTED_CLASS'] = pd.Series(prediction_ids).map(lambda x: TrainingData.ID_TO_CLASS.get(x, "Unknown"))

            output_chunk.to_csv(output_path,
                                mode="w" if not wrote_header else "a",
                                header=not wrote_header,
                                index=False)

            print(f"Wrote {len(output_chunk)} rows to {output_path}. Total rows written: {total_rows_written + len(output_chunk)}")
            total_rows_written += len(output_chunk)

            wrote_header = True

    def inder_quasar_photoz(self):
        pass  # TODO: implement photometric redshift inference

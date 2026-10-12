from pathlib import Path
from typing import Any

import pandas as pd

from kids.catalog.catalog import clean_mask, add_colors, replace_sentinel
from kids.classifiers import XGBClf
from kids.features import COLORS, MAGNITUDES
from kids.training_data import TrainingData


class InferenceCatalog:
    def __init__(
        self,
        model: XGBClf,
        preprocessor: Any,
        index_col: int,
        chunksize: int = 100_000
    ) -> None:
        self.model = model
        self.preprocessor = preprocessor
        self.chunksize = chunksize
        self.index_col = index_col
        self.features = MAGNITUDES + COLORS

    @staticmethod
    def prepare_chunk(chunk: pd.DataFrame) -> pd.DataFrame:

        chunk = clean_mask(chunk)
        chunk = replace_sentinel(chunk)
        chunk = add_colors(chunk)

        return chunk

    def predict_csv(
        self,
        input_path: str | Path,
        output_path: str | Path,
    ) -> None:
        input_path = Path(input_path)
        output_path = Path(output_path)

        wrote_header = False

        for chunk in pd.read_csv(
            input_path,
            chunksize=self.chunksize,
            index_col=self.index_col,
        ):
            prepared_chunk = self.prepare_chunk(chunk)

            if prepared_chunk.empty:
                continue

            features = prepared_chunk.loc[:, self.features]
            transformed_features = self.preprocessor.transform(features)
            prediction_ids = self.model.predict(transformed_features).astype(int)

            output_chunk = chunk.loc[prepared_chunk.index].copy()
            output_chunk["PREDICTED_CLASS_ID"] = prediction_ids
            output_chunk["PREDICTED_CLASS"] = [
                TrainingData.ID_TO_CLASS.get(prediction_id, "UNKNOWN")
                for prediction_id in prediction_ids
            ]

            output_chunk.to_csv(
                output_path,
                mode="w" if not wrote_header else "a",
                header=not wrote_header,
                index=False,
            )

            wrote_header = True

        if not wrote_header:
            pd.DataFrame(
                columns=[
                    "PREDICTED_CLASS_ID",
                    "PREDICTED_CLASS",
                ]
            ).to_csv(output_path, index=False)

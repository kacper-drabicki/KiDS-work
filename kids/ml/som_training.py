from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from joblib import dump
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import RobustScaler

from kids.catalog.catalog import load_and_prepare_catalog
from kids.features import clf_features
from kids.ml.som import MySOM, SOMConfig


@dataclass
class SOMTrainingResult:
    model: MySOM
    preprocessor: Pipeline
    quantization_errors: list[float]
    topographic_errors: list[float]

def make_preprocessor(catalog: pd.DataFrame, config: SOMConfig) -> Pipeline:
    preprocessor = Pipeline(
        [
            ("imputer", SimpleImputer(missing_values=np.nan, strategy="median")),
            ("scaler", RobustScaler()),
        ]
    )
    sample = catalog.loc[:, clf_features].sample(
        n=min(config.preprocessor_sample_size, len(catalog)),
        random_state=config.random_state,
    )
    preprocessor.fit(sample)
    return preprocessor


def train_som(
    config: SOMConfig,
    data_path: Path,
    preprocessor_path: Path,
    model_path: Path,
) -> SOMTrainingResult:
    catalog = load_and_prepare_catalog(data_path)
    preprocessor = make_preprocessor(catalog, config)
    preprocessor_path.parent.mkdir(parents=True, exist_ok=True)
    dump(preprocessor, preprocessor_path)

    features = torch.tensor(
        preprocessor.transform(catalog.loc[:, clf_features]),
        dtype=torch.float32,
    )

    model = MySOM(**config.model_parameters(features.shape[1]))
    model.initialize_weights(data=features, mode=model.initialization_mode)
    quantization_errors, topographic_errors = model.fit(data=features)

    model_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), model_path)

    return SOMTrainingResult(
        model=model,
        preprocessor=preprocessor,
        quantization_errors=quantization_errors,
        topographic_errors=topographic_errors,
    )

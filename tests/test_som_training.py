import numpy as np
import pandas as pd
import torch

from kids.features import clf_features
from kids.ml.som import SOMConfig
from kids.ml.som_training import make_preprocessor, train_som


def make_catalog(rows: int = 8) -> pd.DataFrame:
    values = np.arange(rows * len(clf_features), dtype=float).reshape(
        rows, len(clf_features)
    )
    catalog = pd.DataFrame(values, columns=clf_features)
    catalog.loc[0, clf_features[0]] = np.nan
    return catalog


def test_make_preprocessor_imputes_and_scales_features():
    catalog = make_catalog(rows=3)
    config = SOMConfig(preprocessor_sample_size=2, random_state=7)

    preprocessor = make_preprocessor(catalog, config)
    transformed = preprocessor.transform(catalog.loc[:, clf_features])

    assert transformed.shape == (3, len(clf_features))
    assert np.isfinite(transformed).all()

    sample = catalog.loc[:, clf_features].sample(
        n=2,
        random_state=config.random_state,
    )
    expected_median = sample[clf_features[0]].median()
    assert preprocessor.named_steps["imputer"].statistics_[0] == expected_median


def test_train_som_persists_preprocessor_and_model(monkeypatch, tmp_path):
    catalog = make_catalog()
    monkeypatch.setattr(
        "kids.ml.som_training.load_and_prepare_catalog",
        lambda _: catalog,
    )
    config = SOMConfig(
        preprocessor_sample_size=len(catalog),
        x=2,
        y=2,
        epochs=1,
        batch_size=4,
        device="cpu",
        initialization_mode="random",
        pbc=False,
    )
    preprocessor_path = tmp_path / "nested" / "preprocessor.joblib"
    model_path = tmp_path / "nested" / "model.pt"

    result = train_som(
        config=config,
        data_path=tmp_path / "catalog.csv",
        preprocessor_path=preprocessor_path,
        model_path=model_path,
    )

    assert preprocessor_path.is_file()
    assert model_path.is_file()
    assert result.model.x == config.x
    assert result.model.y == config.y
    assert len(result.quantization_errors) == config.epochs
    assert len(result.topographic_errors) == config.epochs
    saved_state = torch.load(model_path, weights_only=True)
    assert saved_state.keys() == result.model.state_dict().keys()
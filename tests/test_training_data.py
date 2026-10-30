import pytest
import pandas as pd
import numpy as np
from kids.training_data import TrainingData

@pytest.fixture
def data():
    n=100
    return pd.DataFrame({
        "f1": [np.nan if i == 0 else float(i) for i in range(n)],
        "f2": [np.nan if i == 1 else float(2 * i) for i in range(n)],
        "SPECTYPE": ["QSO", "GALAXY", "STAR", "QSO"] * (n // 4),
        "Z": np.arange(n, dtype=float),
    })

def test_make_train_split(data):
    training_data = TrainingData(data, features=["f1", "f2"], train_size=0.8)
    
    assert len(training_data.idx_train) == 80
    assert len(training_data.idx_val) == 10
    assert len(training_data.idx_test) == 10

    assert set(training_data.idx_train).isdisjoint(training_data.idx_val)
    assert set(training_data.idx_train).isdisjoint(training_data.idx_test)
    assert set(training_data.idx_val).isdisjoint(training_data.idx_test)

def test_train_split_is_reproducible(data):
    td1 = TrainingData(data, features=["f1", "f2"])
    td2 = TrainingData(data, features=["f1", "f2"])

    assert all(td1.idx_train == td2.idx_train)
    assert all(td1.idx_val == td2.idx_val)
    assert all(td1.idx_test == td2.idx_test)


def test_preprocessor_is_fitted_only_on_training_data(data):
    training_data = TrainingData(data, features=["f1", "f2"])
    train_features = data.loc[training_data.idx_train, ["f1", "f2"]]

    expected_imputer_statistics = train_features.max()
    imputed_train_features = train_features.fillna(expected_imputer_statistics)
    expected_scaler_min = imputed_train_features.min().to_numpy()
    expected_scaler_max = imputed_train_features.max().to_numpy()

    imputer = training_data.preprocess_features.named_steps["imputer"]
    scaler = training_data.preprocess_features.named_steps["scaler"]

    np.testing.assert_array_equal(imputer.statistics_, expected_imputer_statistics)
    np.testing.assert_array_equal(scaler.data_min_, expected_scaler_min)
    np.testing.assert_array_equal(scaler.data_max_, expected_scaler_max)


def test_class_mapping():
    classes = pd.Series(["STAR", "GALAXY", "QSO", "UNKNOWN"])

    mapped = TrainingData.map_class_to_id(classes)

    pd.testing.assert_series_equal(
        mapped,
        pd.Series([0, 1, 2, np.nan]),
        check_dtype=False,
    )


def test_impute_with_max():
    assert TrainingData.impute_with_max(np.array([1.0, 4.5, 2.0])) == 4.5


@pytest.mark.parametrize("split", ["train", "val", "test"])
def test_classification_getters(data, split):
    training_data = TrainingData(data, features=["f1", "f2"])
    features, target = getattr(training_data, f"get_{split}_clf")()
    indices = getattr(training_data, f"idx_{split}")

    assert features.shape == (len(indices), 2)
    assert np.isfinite(features).all()
    pd.testing.assert_series_equal(
        target,
        data.loc[indices, "SPECTYPE"].map(TrainingData.CLASS_TO_ID),
    )


@pytest.mark.parametrize("split", ["train", "val", "test"])
def test_photoz_getters_return_only_qsos(data, split):
    training_data = TrainingData(data, features=["f1", "f2"])
    features, target = getattr(training_data, f"get_{split}_photoz")()
    indices = getattr(training_data, f"idx_{split}")
    expected = data.loc[indices]
    expected = expected.loc[expected["SPECTYPE"] == "QSO", "Z"].sort_index()

    assert features.shape == (len(expected), 2)
    assert np.isfinite(features).all()
    pd.testing.assert_series_equal(target, expected)

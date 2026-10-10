import numpy as np
import pandas as pd

from kids.classification_data import ClassificationData


def make_data(row_count=20):
    return pd.DataFrame(
        {
            "feature_a": np.arange(row_count, dtype=float),
            "feature_b": [np.nan if index % 5 == 0 else index * 2.0 for index in range(row_count)],
            "SPECTYPE": ["STAR", "GALAXY", "QSO", "STAR"] * (row_count // 4),
        }
    )


def test_splits_are_disjoint_and_cover_every_row():
    data = ClassificationData(make_data(), features=["feature_a", "feature_b"])
    train = set(data.idx_train)
    validation = set(data.idx_val)
    test = set(data.idx_test)
    assert train.isdisjoint(validation)
    assert train.isdisjoint(test)
    assert validation.isdisjoint(test)
    assert train | validation | test == set(range(20))


def test_preprocessing_imputes_and_scales_each_split():
    data = ClassificationData(make_data(), features=["feature_a", "feature_b"])
    X_train, y_train = data.get_train_clf()
    X_validation, y_validation = data.get_val_clf()
    X_test, y_test = data.get_test_clf()
    for features, labels in ((X_train, y_train), (X_validation, y_validation), (X_test, y_test)):
        assert features.shape[1] == 2
        assert features.shape[0] == len(labels)
        assert np.isfinite(features).all()
        assert set(labels.dropna()).issubset({0, 1, 2})


def test_class_mapping_is_reversible_for_known_labels():
    labels = pd.Series(["STAR", "GALAXY", "QSO"])
    ids = ClassificationData.map_class_to_id(labels)
    assert ids.tolist() == [0, 1, 2]
    assert [ClassificationData.ID_TO_CLASS[value] for value in ids] == labels.tolist()

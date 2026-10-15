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
    training_data = TrainingData(data, features=["f1", "f2"])
    
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

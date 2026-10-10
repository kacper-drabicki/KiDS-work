import tempfile
from pathlib import Path

import numpy as np

from kids.classifiers import XGBClf, XGBConfig


def test_model_can_train_predict_and_round_trip():
    X = np.array(
        [
            [0.0, 0.0], [0.0, 1.0], [1.0, 0.0], [1.0, 1.0],
            [0.1, 0.0], [0.0, 0.9], [0.9, 0.0], [0.9, 0.9],
            [0.2, 0.1], [0.1, 0.8], [0.8, 0.1], [0.8, 0.8],
        ],
        dtype=float,
    )
    y = np.array([0, 1, 1, 2, 0, 1, 1, 2, 0, 1, 1, 2])
    config = XGBConfig(n_estimators=5, early_stopping_rounds=2, max_depth=2, n_jobs=1)
    model = XGBClf(config)
    model.train(X[:8], y[:8], X[8:], y[8:], verbose=False)
    predictions = model.predict(X[8:])
    assert predictions.shape == (4,)
    assert np.isin(predictions, [0, 1, 2]).all()
    with tempfile.TemporaryDirectory() as directory:
        path = Path(directory) / "model.json"
        model.save(path)
        loaded = XGBClf()
        loaded.load(path)
        np.testing.assert_array_equal(loaded.predict(X[8:]), predictions)

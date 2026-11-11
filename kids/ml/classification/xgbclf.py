import numpy as np
from dataclasses import asdict
from xgboost import XGBClassifier
from configs.xgbclf_config import XGBConfig

class XGBClf:
    def __init__(self, config: XGBConfig = None):
        self.model = XGBClassifier(**asdict(config)) if config is not None else XGBClassifier()

    def train(self, X_train: np.ndarray, y_train: np.ndarray, X_val: np.ndarray, y_val: np.ndarray, verbose: bool = True) -> None:
        self.model.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=verbose)

    def predict(self, X: np.ndarray) -> np.ndarray:
        return self.model.predict(X)

    def save(self, filepath: str) -> None:
        # XGBoost automatically infers format from extension (.json or .ubj)
        self.model.save_model(filepath)

    def load(self, filepath: str) -> None:
        self.model.load_model(filepath)

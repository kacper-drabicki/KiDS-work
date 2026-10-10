import numpy as np
from dataclasses import dataclass,  asdict
from xgboost import XGBClassifier

#------------------------------------------

@dataclass
class XGBConfig:
    booster: str = 'gbtree'
    objective: str = 'multi:softmax'
    n_estimators: int = 100000
    early_stopping_rounds: int = 20
    max_depth: int = 7
    learning_rate: float = 0.1
    gamma: float = 0.0
    min_child_weight: float = 5.0
    max_delta_step: float = 0.0
    colsample_bytree: float = 0.9
    colsample_bylevel: float = 1.0
    subsample: float = 0.8
    reg_alpha: float = 0.1
    reg_lambda: float = 1.0
    base_score: float = 0.5
    random_state: int = 18235
    missing: float = np.nan
    verbosity: int = 0
    n_jobs: int = 32

#------------------------------------------

class XGBClf:
    def __init__(self, config: XGBConfig = None):
        self.model = XGBClassifier(**asdict(config)) if config is not None else XGBClassifier()

    def train(self, X_train: np.array, y_train: np.array, X_val: np.array, y_val: np.array, verbose: bool = True) -> None:
        self.model.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=verbose)

    def predict(self, X: np.array) -> np.array:
        return self.model.predict(X)

    def save(self, filepath: str) -> None:
        # XGBoost automatically infers format from extension (.json or .ubj)
        self.model.save_model(filepath)

    def load(self, filepath: str) -> None:
        self.model.load_model(filepath)

#------------------------------------------

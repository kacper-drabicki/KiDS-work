import numpy as np
from dataclasses import dataclass

@dataclass
class XGBConfig:
    booster: str = 'gbtree'
    objective: str = 'multi:softmax'
    n_estimators: int = 100000
    early_stopping_rounds: int = 20
    max_depth: int = 9
    learning_rate: float = 0.1
    gamma: float = 0.0
    scale_pos_weight: float = 2.0
    min_child_weight: float = 1.0
    max_delta_step: float = 0.0
    colsample_bytree: float = 0.9
    colsample_bylevel: float = 1.0
    subsample: float = 0.8
    reg_alpha: float = 0.0
    reg_lambda: float = 1.0
    base_score: float = 0.5
    random_state: int = 18235
    missing: float = np.nan
    verbosity: int = 0
    n_jobs: int = 16

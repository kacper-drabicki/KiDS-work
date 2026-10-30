import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import RobustScaler

def get_preprocessor(df: pd.DataFrame, n_samples_for_preprocessor: int) -> Pipeline:
    preprocessor = Pipeline(
        [
            ("imputer", SimpleImputer(missing_values=np.nan, strategy="median")),
            ("scaler", RobustScaler()),
        ])
    preprocessor.fit(df.sample(n_samples_for_preprocessor, random_state=42))
    return preprocessor




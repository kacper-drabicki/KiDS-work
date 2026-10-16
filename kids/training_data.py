'''
    Acknowledgement: The code below is heavily inspired by the work of snakoneczny.
'''

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import MinMaxScaler
from sklearn.pipeline import Pipeline
from kids.features import MAGNITUDES, COLORS, RATIOS


class TrainingData:

    CLASS_TO_ID = {'QSO': 2, 'GALAXY': 1, 'STAR': 0}
    ID_TO_CLASS = {v: k for k, v in CLASS_TO_ID.items()}

    @staticmethod
    def map_class_to_id(y: pd.Series) -> pd.Series:
        return y.map(TrainingData.CLASS_TO_ID)

    @staticmethod
    def impute_with_max(values: np.ndarray) -> float:
        return float(np.max(values))

    def __init__(self, data: pd.DataFrame, features: list[str]) -> None:
        self.data = data.reset_index(drop=True, inplace=False)
        self.features = features
        self.clf_target = 'SPECTYPE'
        self.preprocess_clf_target = self.map_class_to_id
        self.photoz_target = 'Z'
        
        self.make_train_split()
        self.fit_preprocessor()

    def make_train_split(self) -> None:
        idx = np.arange(self.data.shape[0])
        self.idx_train, self.idx_test = train_test_split(idx, test_size = 0.2, random_state=42)
        self.idx_val, self.idx_test = train_test_split(self.idx_test, test_size=0.5, random_state=42)

    def fit_preprocessor(self) -> None:
        X_train = self.data.loc[self.idx_train, self.features]
        self.preprocess_features = Pipeline([('imputer', SimpleImputer(missing_values=np.nan, strategy=self.impute_with_max)),
                                      ('scaler', MinMaxScaler())])
        self.preprocess_features.fit(X_train)

    def get_train_clf(self) -> tuple[np.ndarray, pd.Series]:
        X_train, y_train = self.data.loc[self.idx_train, self.features], self.data.loc[self.idx_train, self.clf_target]

        return self.preprocess_features.transform(X_train), self.preprocess_clf_target(y_train)

    def get_val_clf(self) -> tuple[np.ndarray, pd.Series]:
        X_val, y_val = self.data.loc[self.idx_val, self.features], self.data.loc[self.idx_val, self.clf_target]
        
        return self.preprocess_features.transform(X_val), self.preprocess_clf_target(y_val)

    def get_test_clf(self) -> tuple[np.ndarray, pd.Series]:
        X_test, y_test = self.data.loc[self.idx_test, self.features], self.data.loc[self.idx_test, self.clf_target]

        return self.preprocess_features.transform(X_test), self.preprocess_clf_target(y_test)

    def get_train_photoz(self) -> tuple[np.ndarray, pd.Series]:
        idx_train = (self.data.index.isin(self.idx_train)) & (self.data['SPECTYPE'] == 'QSO')
        X_train, y_train = self.data.loc[idx_train, self.features], self.data.loc[idx_train, self.photoz_target]

        return self.preprocess_features.transform(X_train), y_train

    def get_val_photoz(self) -> tuple[np.ndarray, pd.Series]:
        idx_val = (self.data.index.isin(self.idx_val)) & (self.data['SPECTYPE'] == 'QSO')
        X_val, y_val = self.data.loc[idx_val, self.features], self.data.loc[idx_val, self.photoz_target]

        return self.preprocess_features.transform(X_val), y_val

    def get_test_photoz(self) -> tuple[np.ndarray, pd.Series]:
        idx_test = (self.data.index.isin(self.idx_test)) & (self.data['SPECTYPE'] == 'QSO')
        X_test, y_test = self.data.loc[idx_test, self.features], self.data.loc[idx_test, self.photoz_target]

        return self.preprocess_features.transform(X_test), y_test

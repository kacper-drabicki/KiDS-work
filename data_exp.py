'''
    Acknowledgement: The code below is heavily inspired by the work of snakoneczny.
'''

import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import MinMaxScaler
from sklearn.pipeline import Pipeline
from magnitudes_and_colors import MAGNITUDES, COLORS


class ExperimentalData():

    CLASS_TO_ID = {'QSO': 2, 'GALAXY': 1, 'STAR': 0}
    ID_TO_CLASS = {v: k for k, v in CLASS_TO_ID.items()}

    def __init__(self, data, features = MAGNITUDES + COLORS):
        self.data = data.reset_index(drop=True, inplace=False)
        self.features = features
        self.clf_target = 'SPECTYPE'
        self.preprocess_target = lambda y: y.map(self.CLASS_TO_ID)
        
        self.make_train_split()
        self.fit_preprocessor()

    def make_train_split(self):
        idx = np.arange(self.data.shape[0])
        self.idx_train, self.idx_test = train_test_split(idx, test_size = 0.2, random_state=42)
        self.idx_val, self.idx_test = train_test_split(self.idx_test, test_size=0.5, random_state=42)

    def fit_preprocessor(self):
        X_train = self.data.loc[self.idx_train, self.features]
        self.preprocess_features = Pipeline([('imputer', SimpleImputer(missing_values=np.nan, strategy=lambda x: np.max(x))),
                                      ('scaler', MinMaxScaler())])
        self.preprocess_features.fit(X_train)

    def get_train_clf(self):
        X_train, y_train = self.data.loc[self.idx_train, self.features], self.data.loc[self.idx_train, self.clf_target]

        return self.preprocess_features.transform(X_train), self.preprocess_target(y_train)

    def get_val_clf(self):
        X_val, y_val = self.data.loc[self.idx_val, self.features], self.data.loc[self.idx_val, self.clf_target]
        
        return self.preprocess_features.transform(X_val), self.preprocess_target(y_val)

    def get_test_clf(self):
        X_test, y_test = self.data.loc[self.idx_test, self.features], self.data.loc[self.idx_test, self.clf_target]

        return self.preprocess_features.transform(X_test), self.preprocess_target(y_test)

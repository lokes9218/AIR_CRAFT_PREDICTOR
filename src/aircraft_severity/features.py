from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import StandardScaler
import pandas as pd
import numpy as np

class AircraftFeatureScaler(BaseEstimator, TransformerMixin):
    """
    Custom transformer wrapping StandardScaler to preserve pandas DataFrame column names and handle missing values cleanly.
    """
    def __init__(self):
        self.scaler = StandardScaler()
        self.feature_names_in_ = None

    def fit(self, X, y=None):
        if isinstance(X, pd.DataFrame):
            self.feature_names_in_ = X.columns.tolist()
        self.scaler.fit(X)
        return self

    def transform(self, X):
        scaled_array = self.scaler.transform(X)
        if self.feature_names_in_ is not None and isinstance(X, pd.DataFrame):
            return pd.DataFrame(scaled_array, columns=self.feature_names_in_, index=X.index)
        return scaled_array

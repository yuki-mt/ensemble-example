from sklearn.base import BaseEstimator, ClassifierMixin
import pandas as pd
import numpy as np

class MyClassifier(BaseEstimator, ClassifierMixin):
    def __init__(self, threshold: float = 10.0):
        self.threshold = threshold

    def fit(self, X, y):
        return self

    def _is_greater(self, x):
        return x.mean() > self.threshold

    def predict(self, X):
        X = X.to_numpy() if isinstance(X, pd.DataFrame) else X
        return np.array([self._is_greater(x) for x in X])

    def score(self, X):
        return sum(self.predict(X))

import json
from pathlib import Path
import joblib
import numpy as np
import pandas as pd
from sklearn.feature_selection import mutual_info_classif
from sklearn.preprocessing import MinMaxScaler

class LeakageFreePreprocessor:
    def __init__(self, correlation_threshold=0.95, top_k_features=40, random_state=42):
        self.correlation_threshold = correlation_threshold
        self.top_k_features = top_k_features
        self.random_state = random_state
        self.medians_ = None
        self.kept_after_corr_ = None
        self.selected_features_ = None
        self.scaler_ = MinMaxScaler()

    def fit(self, X: pd.DataFrame, y):
        X = X.copy()
        self.medians_ = X.median(numeric_only=True).to_dict()
        X = self._impute(X)
        corr = X.corr(numeric_only=True).abs()
        upper = corr.where(np.triu(np.ones(corr.shape), k=1).astype(bool))
        drop = [c for c in upper.columns if (upper[c] > self.correlation_threshold).any()]
        self.kept_after_corr_ = [c for c in X.columns if c not in drop]
        Xc = X[self.kept_after_corr_]
        k = min(self.top_k_features, Xc.shape[1])
        if k < Xc.shape[1]:
            mi = mutual_info_classif(Xc, y, random_state=self.random_state)
            order = np.argsort(mi)[::-1][:k]
            self.selected_features_ = [Xc.columns[i] for i in order]
        else:
            self.selected_features_ = list(Xc.columns)
        self.scaler_.fit(X[self.selected_features_])
        return self

    def _impute(self, X):
        X = X.copy()
        for c in X.columns:
            med = self.medians_.get(c, 0.0) if self.medians_ else 0.0
            X[c] = pd.to_numeric(X[c], errors='coerce').fillna(med)
        return X

    def transform(self, X: pd.DataFrame):
        missing = [c for c in self.selected_features_ if c not in X.columns]
        if missing:
            raise ValueError(f'Missing selected features for frozen preprocessing: {missing}')
        X = self._impute(X[self.selected_features_])
        arr = self.scaler_.transform(X)
        return pd.DataFrame(arr, columns=self.selected_features_, index=X.index)

    def fit_transform(self, X, y):
        return self.fit(X, y).transform(X)

    def save(self, path):
        p = Path(path); p.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(self, p)

    @staticmethod
    def load(path): return joblib.load(path)

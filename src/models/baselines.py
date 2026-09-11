"""Trivial classification baselines for churn prediction."""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.utils.validation import check_is_fitted


class MajorityClassBaseline(BaseEstimator, ClassifierMixin):
    """Always predict the majority class observed in training."""

    def fit(self, X, y):
        y = np.asarray(y)
        values, counts = np.unique(y, return_counts=True)
        self.classes_ = values
        self.majority_class_ = values[np.argmax(counts)]
        # Probability of positive class for predict_proba compatibility
        pos_rate = float(np.mean(y == 1)) if 1 in values else 0.0
        self.pos_rate_ = pos_rate
        return self

    def predict(self, X):
        check_is_fitted(self, "majority_class_")
        n = len(X) if hasattr(X, "__len__") else X.shape[0]
        return np.full(n, self.majority_class_, dtype=int)

    def predict_proba(self, X):
        check_is_fitted(self, "pos_rate_")
        n = len(X) if hasattr(X, "__len__") else X.shape[0]
        p1 = np.full(n, self.pos_rate_)
        return np.column_stack([1 - p1, p1])


class SimpleRuleBaseline(BaseEstimator, ClassifierMixin):
    """Predict churn for month-to-month contracts with above-median monthly charges.

    A transparent, domain-inspired rule that should beat (or at least differ from)
    the majority-class baseline on recall.
    """

    def fit(self, X: pd.DataFrame, y):
        if not isinstance(X, pd.DataFrame):
            raise TypeError("SimpleRuleBaseline expects a pandas DataFrame.")
        self.charge_threshold_ = float(X["MonthlyCharges"].median())
        self.classes_ = np.array([0, 1])
        return self

    def _rule(self, X: pd.DataFrame) -> np.ndarray:
        check_is_fitted(self, "charge_threshold_")
        month_to_month = X["Contract"].astype(str).str.lower().eq("month-to-month")
        high_charges = X["MonthlyCharges"] >= self.charge_threshold_
        return (month_to_month & high_charges).astype(int).to_numpy()

    def predict(self, X: pd.DataFrame):
        return self._rule(X)

    def predict_proba(self, X: pd.DataFrame):
        preds = self._rule(X).astype(float)
        # Hard 0/1 probabilities for the rule
        return np.column_stack([1 - preds, preds])

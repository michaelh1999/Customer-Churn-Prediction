"""Tests for classification metrics."""

import numpy as np

from src.evaluation.metrics import classification_metrics


def test_perfect_predictions():
    y = np.array([0, 0, 1, 1])
    proba = np.array([0.1, 0.2, 0.9, 0.8])
    m = classification_metrics(y, y_proba=proba, threshold=0.5)
    assert m["accuracy"] == 1.0
    assert m["precision"] == 1.0
    assert m["recall"] == 1.0
    assert m["f1"] == 1.0
    assert m["confusion_matrix"]["tp"] == 2
    assert m["confusion_matrix"]["tn"] == 2


def test_majority_like_zero_recall():
    y = np.array([0, 0, 0, 1])
    pred = np.array([0, 0, 0, 0])
    m = classification_metrics(y, y_pred=pred)
    assert m["recall"] == 0.0
    assert m["precision"] == 0.0

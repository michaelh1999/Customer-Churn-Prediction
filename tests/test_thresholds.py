"""Tests for threshold sweeps."""

import numpy as np

from src.evaluation.thresholds import best_threshold, sweep_thresholds, threshold_metrics


def test_threshold_metrics_counts():
    y = np.array([0, 0, 1, 1])
    p = np.array([0.1, 0.6, 0.4, 0.9])
    m = threshold_metrics(y, p, 0.5)
    assert m["tp"] == 1
    assert m["fp"] == 1
    assert m["tn"] == 1
    assert m["fn"] == 1
    assert abs(m["positive_prediction_rate"] - 0.5) < 1e-9


def test_sweep_monotonic_positive_rate():
    y = np.array([0, 1, 0, 1, 0, 1])
    p = np.array([0.1, 0.2, 0.4, 0.6, 0.8, 0.9])
    sweep = sweep_thresholds(y, p, min_t=0.1, max_t=0.9, step=0.2)
    assert list(sweep["threshold"]) == [0.1, 0.3, 0.5, 0.7, 0.9]
    # Higher threshold => fewer or equal positives
    rates = sweep["positive_prediction_rate"].tolist()
    assert rates == sorted(rates, reverse=True)


def test_best_threshold_f1():
    y = np.array([0, 0, 1, 1])
    p = np.array([0.2, 0.4, 0.6, 0.8])
    sweep = sweep_thresholds(y, p, min_t=0.3, max_t=0.7, step=0.2)
    t = best_threshold(sweep, "f1")
    assert t in set(sweep["threshold"])

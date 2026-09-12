"""Tests for business value calculations."""

from src.business.costs import add_business_value, expected_value
from src.evaluation.thresholds import sweep_thresholds
import numpy as np
import pandas as pd


def test_expected_value_formula():
    # TP=2 -> +400, FP=1 -> -50, FN=1 -> -500 => -150
    assert expected_value(2, 1, 5, 1, 50.0, 500.0, 200.0) == 2 * 200 - 50 - 500


def test_add_business_value_column():
    y = np.array([0, 0, 1, 1])
    p = np.array([0.2, 0.6, 0.4, 0.9])
    sweep = sweep_thresholds(y, p, min_t=0.5, max_t=0.5, step=0.05)
    valued = add_business_value(
        sweep,
        false_positive_cost=50,
        false_negative_cost=500,
        retention_value=200,
    )
    assert "business_value" in valued.columns
    row = valued.iloc[0]
    assert row["business_value"] == expected_value(
        int(row.tp), int(row.fp), int(row.tn), int(row.fn), 50, 500, 200
    )

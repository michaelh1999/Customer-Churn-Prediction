"""Business cost and expected-value helpers for churn decisions."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd

from src.data.load import load_config, project_root
from src.evaluation.thresholds import best_threshold, sweep_thresholds


def expected_value(
    tp: int,
    fp: int,
    tn: int,
    fn: int,
    false_positive_cost: float,
    false_negative_cost: float,
    retention_value: float,
) -> float:
    """Expected net business value for a confusion matrix.

    Assumptions (educational):
    - Each true positive yields ``retention_value`` (successful save) minus outreach cost
      already captured separately? We treat FP cost as outreach waste and TP as net retention
      value after outreach (retention_value is net of offer cost).
    - Each false positive costs ``false_positive_cost`` (wasted offer).
    - Each false negative costs ``false_negative_cost`` (lost customer, no intervention).
    - True negatives contribute 0 incremental cost/value.
    """
    return (
        tp * retention_value
        - fp * false_positive_cost
        - fn * false_negative_cost
    )


def add_business_value(
    sweep_df: pd.DataFrame,
    false_positive_cost: float | None = None,
    false_negative_cost: float | None = None,
    retention_value: float | None = None,
    config: dict[str, Any] | None = None,
) -> pd.DataFrame:
    """Attach expected business value column to a threshold sweep DataFrame."""
    cfg = config or load_config()
    biz = cfg["business"]
    fp_cost = false_positive_cost if false_positive_cost is not None else biz["false_positive_cost"]
    fn_cost = false_negative_cost if false_negative_cost is not None else biz["false_negative_cost"]
    ret_val = retention_value if retention_value is not None else biz["retention_value"]

    out = sweep_df.copy()
    out["business_value"] = [
        expected_value(
            int(r.tp),
            int(r.fp),
            int(r.tn),
            int(r.fn),
            fp_cost,
            fn_cost,
            ret_val,
        )
        for r in out.itertuples()
    ]
    out.attrs["business_assumptions"] = {
        "false_positive_cost": fp_cost,
        "false_negative_cost": fn_cost,
        "retention_value": ret_val,
    }
    return out


def compare_operating_points(
    sweep_with_value: pd.DataFrame,
) -> dict[str, Any]:
    """Compare default 0.5, F1-optimal, and business-value-optimal thresholds."""
    def row_at(threshold: float) -> dict[str, Any]:
        # nearest threshold in sweep
        idx = (sweep_with_value["threshold"] - threshold).abs().idxmin()
        r = sweep_with_value.loc[idx]
        return {
            "threshold": float(r["threshold"]),
            "f1": float(r["f1"]),
            "precision": float(r["precision"]),
            "recall": float(r["recall"]),
            "accuracy": float(r["accuracy"]),
            "business_value": float(r["business_value"]),
            "tp": int(r["tp"]),
            "fp": int(r["fp"]),
            "tn": int(r["tn"]),
            "fn": int(r["fn"]),
        }

    f1_t = best_threshold(sweep_with_value, "f1")
    biz_t = best_threshold(sweep_with_value, "business_value")

    comparison = {
        "default_0_5": row_at(0.5),
        "f1_optimal": row_at(f1_t),
        "business_value_optimal": row_at(biz_t),
        "assumptions": sweep_with_value.attrs.get("business_assumptions", {}),
        "recommendation": {
            "threshold": biz_t,
            "rationale": (
                "Prefer the business-value-optimal threshold when FP/FN costs and "
                "retention value are agreed assumptions; F1 treats errors symmetrically."
            ),
        },
    }
    return comparison


def run_business_analysis(model_name: str = "logistic_regression") -> dict[str, Any]:
    """Load threshold sweep (or rebuild), score business value, persist report."""
    root = project_root()
    cfg = load_config()
    sweep_path = root / cfg["paths"]["reports_dir"] / f"{model_name}_threshold_sweep.csv"
    if sweep_path.exists():
        sweep = pd.read_csv(sweep_path)
    else:
        proba_path = root / cfg["paths"]["processed_dir"] / f"{model_name}_test_proba.csv"
        proba_df = pd.read_csv(proba_path)
        sweep = sweep_thresholds(proba_df["y_true"], proba_df["y_proba"])

    valued = add_business_value(sweep, config=cfg)
    valued_path = root / cfg["paths"]["reports_dir"] / f"{model_name}_threshold_business.csv"
    valued.to_csv(valued_path, index=False)

    comparison = compare_operating_points(valued)
    out_json = root / cfg["paths"]["reports_dir"] / f"{model_name}_business_thresholds.json"
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(comparison, f, indent=2)

    print(json.dumps(comparison, indent=2))
    print(f"Wrote {valued_path}")
    print(f"Wrote {out_json}")
    return comparison

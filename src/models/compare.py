"""Compare linear vs nonlinear churn models."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd

from src.business.costs import add_business_value, compare_operating_points
from src.data.load import load_config, project_root
from src.evaluation.metrics import metrics_to_row
from src.evaluation.thresholds import sweep_thresholds
from src.models.train import train_and_evaluate


def compare_models() -> dict[str, Any]:
    """Train LR + HistGradientBoosting and write a side-by-side report."""
    root = project_root()
    cfg = load_config()

    lr = train_and_evaluate("logistic_regression", cfg)
    hgb = train_and_evaluate("hist_gradient_boosting", cfg)

    rows = [
        metrics_to_row("logistic_regression", lr["test_metrics"]),
        metrics_to_row("hist_gradient_boosting", hgb["test_metrics"]),
    ]
    table = pd.DataFrame(rows)

    # Business value at each model's own BV-optimal threshold
    business: dict[str, Any] = {}
    for name in ("logistic_regression", "hist_gradient_boosting"):
        proba_path = root / cfg["paths"]["processed_dir"] / f"{name}_test_proba.csv"
        proba_df = pd.read_csv(proba_path)
        sweep = add_business_value(sweep_thresholds(proba_df["y_true"], proba_df["y_proba"]), config=cfg)
        business[name] = compare_operating_points(sweep)

    report = {
        "metrics_table": rows,
        "business_by_model": business,
        "tradeoffs": {
            "logistic_regression": {
                "predictive_performance": "Strong linear baseline; competitive ROC/PR-AUC.",
                "interpretability": "High — coefficients relate to log-odds.",
                "complexity": "Low — fast train/infer, few hyperparameters.",
                "training_inference_cost": "Very low.",
            },
            "hist_gradient_boosting": {
                "predictive_performance": "Can capture nonlinear interactions without heavy tuning.",
                "interpretability": "Lower — need importances/SHAP for explanations.",
                "complexity": "Medium — trees, more knobs, larger artifact.",
                "training_inference_cost": "Higher than LR but still fine for this dataset size.",
            },
        },
    }

    out = root / cfg["paths"]["reports_dir"] / "model_comparison.json"
    with open(out, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)
    table.to_csv(root / cfg["paths"]["reports_dir"] / "model_comparison.csv", index=False)
    print(table.to_string(index=False))
    print(f"Wrote {out}")
    return report


if __name__ == "__main__":
    compare_models()

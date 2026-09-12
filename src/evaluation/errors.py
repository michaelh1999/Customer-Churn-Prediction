"""Error analysis: segment TP/TN/FP/FN by key features."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from src.data.load import load_config, project_root
from src.data.split import load_split
from src.models.train import load_model, prepare_xy


def assign_error_types(y_true, y_pred) -> np.ndarray:
    """Return labels in {TP, TN, FP, FN}."""
    y_true = np.asarray(y_true).astype(int)
    y_pred = np.asarray(y_pred).astype(int)
    out = np.empty(len(y_true), dtype=object)
    out[(y_true == 1) & (y_pred == 1)] = "TP"
    out[(y_true == 0) & (y_pred == 0)] = "TN"
    out[(y_true == 0) & (y_pred == 1)] = "FP"
    out[(y_true == 1) & (y_pred == 0)] = "FN"
    return out


def error_analysis(
    model_name: str = "logistic_regression",
    threshold: float | None = None,
) -> dict:
    """Segment test-set errors by Contract, tenure bins, and MonthlyCharges."""
    root = project_root()
    cfg = load_config()
    threshold = float(threshold if threshold is not None else cfg["model"]["threshold"])

    _, test_df = load_split()
    bundle = load_model(model_name, cfg)
    pipe = bundle["pipeline"]
    X_test, y_test = prepare_xy(test_df)
    proba = pipe.predict_proba(X_test)[:, 1]
    pred = (proba >= threshold).astype(int)
    errors = assign_error_types(y_test, pred)

    analysis = test_df.copy()
    analysis["y_true"] = y_test.to_numpy()
    analysis["y_proba"] = proba
    analysis["y_pred"] = pred
    analysis["error_type"] = errors
    analysis["tenure_bin"] = pd.cut(
        analysis["tenure"],
        bins=[-0.1, 12, 24, 48, 72],
        labels=["0-12", "13-24", "25-48", "49-72"],
    )
    analysis["charges_bin"] = pd.qcut(
        analysis["MonthlyCharges"], q=4, labels=["Q1", "Q2", "Q3", "Q4"], duplicates="drop"
    )

    def crosstab(col: str) -> dict:
        ct = pd.crosstab(analysis[col], analysis["error_type"], normalize="index")
        return ct.round(3).to_dict()

    summary = {
        "model_name": model_name,
        "threshold": threshold,
        "counts": analysis["error_type"].value_counts().to_dict(),
        "by_contract": crosstab("Contract"),
        "by_tenure_bin": crosstab("tenure_bin"),
        "by_charges_bin": crosstab("charges_bin"),
        "fn_profile": {
            "mean_tenure": float(analysis.loc[analysis["error_type"] == "FN", "tenure"].mean())
            if (analysis["error_type"] == "FN").any()
            else None,
            "top_contracts": analysis.loc[analysis["error_type"] == "FN", "Contract"]
            .value_counts()
            .head(5)
            .to_dict(),
        },
        "fp_profile": {
            "mean_tenure": float(analysis.loc[analysis["error_type"] == "FP", "tenure"].mean())
            if (analysis["error_type"] == "FP").any()
            else None,
            "top_contracts": analysis.loc[analysis["error_type"] == "FP", "Contract"]
            .value_counts()
            .head(5)
            .to_dict(),
        },
        "notes": [
            "False negatives (missed churners) often look like longer-tenure or longer-contract customers who still leave.",
            "False positives often concentrate in month-to-month / high-charge segments that look risky but stay.",
            "Feature gaps: no support tickets, usage drops, or payment failures — common real-world churn signals.",
        ],
    }

    out_path = root / cfg["paths"]["reports_dir"] / f"{model_name}_error_analysis.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    detail_path = root / cfg["paths"]["reports_dir"] / f"{model_name}_error_rows.csv"
    analysis[
        ["customerID", "Contract", "tenure", "MonthlyCharges", "y_true", "y_proba", "y_pred", "error_type"]
    ].to_csv(detail_path, index=False)

    print(json.dumps({k: summary[k] for k in ("counts", "fn_profile", "fp_profile", "notes")}, indent=2))
    print(f"Wrote {out_path}")
    return summary


if __name__ == "__main__":
    error_analysis()

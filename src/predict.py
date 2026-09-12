"""CLI inference for a single customer JSON record."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import pandas as pd

from src.data.load import load_config, project_root
from src.models.train import load_model


FEATURE_COLUMNS = [
    "gender",
    "SeniorCitizen",
    "Partner",
    "Dependents",
    "tenure",
    "PhoneService",
    "MultipleLines",
    "InternetService",
    "OnlineSecurity",
    "OnlineBackup",
    "DeviceProtection",
    "TechSupport",
    "StreamingTV",
    "StreamingMovies",
    "Contract",
    "PaperlessBilling",
    "PaymentMethod",
    "MonthlyCharges",
    "TotalCharges",
]


def customer_to_frame(customer: dict[str, Any]) -> pd.DataFrame:
    """Build a one-row DataFrame aligned with training features."""
    row = {col: customer.get(col) for col in FEATURE_COLUMNS}
    df = pd.DataFrame([row])
    if "TotalCharges" in df.columns:
        df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")
    if "SeniorCitizen" in df.columns:
        df["SeniorCitizen"] = pd.to_numeric(df["SeniorCitizen"], errors="coerce").fillna(0).astype(int)
    if "tenure" in df.columns:
        df["tenure"] = pd.to_numeric(df["tenure"], errors="coerce")
    if "MonthlyCharges" in df.columns:
        df["MonthlyCharges"] = pd.to_numeric(df["MonthlyCharges"], errors="coerce")
    return df


def recommend_action(will_churn: bool) -> str:
    if will_churn:
        return "offer retention"
    return "no action"


def predict_customer(
    customer: dict[str, Any],
    config: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Return churn probability, thresholded decision, and recommended action."""
    cfg = config or load_config()
    model_name = cfg["model"]["selected"]
    threshold = float(cfg["model"]["threshold"])
    bundle = load_model(model_name, cfg)
    pipe = bundle["pipeline"]
    X = customer_to_frame(customer)
    proba = float(pipe.predict_proba(X)[0, 1])
    will_churn = proba >= threshold
    return {
        "churn_probability": round(proba, 4),
        "threshold": threshold,
        "churn_prediction": bool(will_churn),
        "recommended_action": recommend_action(will_churn),
        "model": model_name,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Predict customer churn from a JSON record.")
    parser.add_argument(
        "--input",
        required=True,
        help="Path to a JSON file with one customer record",
    )
    parser.add_argument(
        "--config",
        default=None,
        help="Optional path to YAML config (defaults to configs/default.yaml)",
    )
    args = parser.parse_args(argv)

    root = project_root()
    input_path = Path(args.input)
    if not input_path.is_absolute():
        input_path = (Path.cwd() / input_path).resolve()
        if not input_path.exists():
            input_path = root / args.input

    with open(input_path, encoding="utf-8") as f:
        customer = json.load(f)

    cfg = load_config(args.config) if args.config else load_config()
    result = predict_customer(customer, cfg)
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

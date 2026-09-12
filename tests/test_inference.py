"""Tests for inference CLI helpers."""

import json
from pathlib import Path

from src.predict import customer_to_frame, predict_customer, recommend_action


def test_recommend_action():
    assert recommend_action(True) == "offer retention"
    assert recommend_action(False) == "no action"


def test_customer_to_frame_dtypes():
    customer = {
        "gender": "Female",
        "SeniorCitizen": 0,
        "Partner": "Yes",
        "Dependents": "No",
        "tenure": 2,
        "PhoneService": "Yes",
        "MultipleLines": "No",
        "InternetService": "Fiber optic",
        "OnlineSecurity": "No",
        "OnlineBackup": "No",
        "DeviceProtection": "No",
        "TechSupport": "No",
        "StreamingTV": "No",
        "StreamingMovies": "No",
        "Contract": "Month-to-month",
        "PaperlessBilling": "Yes",
        "PaymentMethod": "Electronic check",
        "MonthlyCharges": 85.5,
        "TotalCharges": "171.0",
    }
    df = customer_to_frame(customer)
    assert len(df) == 1
    assert df["TotalCharges"].dtype.kind == "f"


def test_predict_customer_smoke(trained_lr, config):
    example = Path(__file__).resolve().parents[1] / "examples" / "customer.json"
    with open(example, encoding="utf-8") as f:
        customer = json.load(f)
    result = predict_customer(customer, config)
    assert "churn_probability" in result
    assert 0.0 <= result["churn_probability"] <= 1.0
    assert result["threshold"] == config["model"]["threshold"]
    assert isinstance(result["churn_prediction"], bool)
    assert result["recommended_action"] in {"offer retention", "no action"}

"""Train and persist churn classification models."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import joblib
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

from src.data.load import load_config, load_raw_data, project_root
from src.data.split import make_stratified_split, save_split
from src.data.validate import prepare_target
from src.evaluation.metrics import classification_metrics, metrics_to_row
from src.features.preprocess import build_preprocessor, get_feature_lists


FEATURE_DROP = ("customerID", "Churn")


def prepare_xy(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    X = df.drop(columns=list(FEATURE_DROP), errors="ignore")
    y = df["Churn"].astype(int)
    return X, y


def build_logistic_pipeline(df_for_schema: pd.DataFrame, config: dict[str, Any] | None = None) -> Pipeline:
    cfg = config or load_config()
    params = cfg["model"]["logistic_regression"]
    pre = build_preprocessor(df_for_schema)
    clf = LogisticRegression(
        max_iter=params.get("max_iter", 1000),
        class_weight=params.get("class_weight", "balanced"),
        C=params.get("C", 1.0),
        solver="lbfgs",
    )
    return Pipeline([("preprocess", pre), ("clf", clf)])


def build_hgb_pipeline(df_for_schema: pd.DataFrame, config: dict[str, Any] | None = None) -> Pipeline:
    cfg = config or load_config()
    params = cfg["model"]["hist_gradient_boosting"]
    pre = build_preprocessor(df_for_schema)
    clf = HistGradientBoostingClassifier(
        max_iter=params.get("max_iter", 100),
        learning_rate=params.get("learning_rate", 0.1),
        max_depth=params.get("max_depth", 6),
        random_state=params.get("random_state", 42),
        class_weight="balanced",
    )
    return Pipeline([("preprocess", pre), ("clf", clf)])


def train_and_evaluate(
    model_name: str = "logistic_regression",
    config: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Train a named model, evaluate on holdout, save artifact + metrics."""
    cfg = config or load_config()
    root = project_root()

    df = prepare_target(load_raw_data())
    train_df, test_df = make_stratified_split(df, cfg)
    save_split(train_df, test_df)

    X_train, y_train = prepare_xy(train_df)
    X_test, y_test = prepare_xy(test_df)

    if model_name == "logistic_regression":
        pipe = build_logistic_pipeline(train_df, cfg)
    elif model_name in ("hist_gradient_boosting", "hgb"):
        model_name = "hist_gradient_boosting"
        pipe = build_hgb_pipeline(train_df, cfg)
    else:
        raise ValueError(f"Unknown model_name: {model_name}")

    pipe.fit(X_train, y_train)

    # Training loss (log loss on train probabilities) vs holdout metrics
    train_proba = pipe.predict_proba(X_train)[:, 1]
    test_proba = pipe.predict_proba(X_test)[:, 1]
    train_metrics = classification_metrics(y_train, y_proba=train_proba)
    test_metrics = classification_metrics(y_test, y_proba=test_proba)

    models_dir = root / cfg["paths"]["models_dir"]
    models_dir.mkdir(parents=True, exist_ok=True)
    model_path = models_dir / f"{model_name}.joblib"
    joblib.dump(
        {
            "pipeline": pipe,
            "model_name": model_name,
            "numeric_features": get_feature_lists(train_df)[0],
            "categorical_features": get_feature_lists(train_df)[1],
        },
        model_path,
    )

    # Persist test probabilities for later threshold / calibration notebooks
    proba_path = root / cfg["paths"]["processed_dir"] / f"{model_name}_test_proba.csv"
    pd.DataFrame(
        {
            "y_true": y_test.to_numpy(),
            "y_proba": test_proba,
        }
    ).to_csv(proba_path, index=False)

    result = {
        "model_name": model_name,
        "model_path": str(model_path.relative_to(root)),
        "train_metrics": train_metrics,
        "test_metrics": test_metrics,
        "n_train": len(train_df),
        "n_test": len(test_df),
    }

    reports_dir = root / cfg["paths"]["reports_dir"]
    reports_dir.mkdir(parents=True, exist_ok=True)
    out_json = reports_dir / f"{model_name}_metrics.json"
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2)

    print(metrics_to_row(model_name, test_metrics))
    print(f"Saved model -> {model_path}")
    print(f"Saved metrics -> {out_json}")
    return result


def load_model(model_name: str | None = None, config: dict[str, Any] | None = None):
    """Load a saved model bundle from reports/models/."""
    cfg = config or load_config()
    root = project_root()
    name = model_name or cfg["model"]["selected"]
    path = root / cfg["paths"]["models_dir"] / f"{name}.joblib"
    if not path.exists():
        raise FileNotFoundError(f"Model not found at {path}. Train it first.")
    return joblib.load(path)


if __name__ == "__main__":
    train_and_evaluate("logistic_regression")

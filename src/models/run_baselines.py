"""Run and persist classification baselines."""

from __future__ import annotations

import json
from pathlib import Path

from src.data.load import load_config, project_root
from src.data.split import load_split, make_stratified_split, save_split
from src.data.validate import prepare_target
from src.data.load import load_raw_data
from src.evaluation.metrics import classification_metrics, metrics_to_row
from src.models.baselines import MajorityClassBaseline, SimpleRuleBaseline


FEATURE_DROP = ("customerID", "Churn")


def _xy(df):
    X = df.drop(columns=list(FEATURE_DROP), errors="ignore")
    y = df["Churn"].astype(int)
    return X, y


def run_baselines() -> dict:
    """Fit baselines on train, evaluate on test, write reports/baselines.json."""
    cfg = load_config()
    root = project_root()

    df = prepare_target(load_raw_data())
    train_df, test_df = make_stratified_split(df, cfg)
    save_split(train_df, test_df)

    X_train, y_train = _xy(train_df)
    X_test, y_test = _xy(test_df)

    majority = MajorityClassBaseline().fit(X_train, y_train)
    rule = SimpleRuleBaseline().fit(X_train, y_train)

    results = {
        "majority_class": classification_metrics(
            y_test, y_pred=majority.predict(X_test), y_proba=majority.predict_proba(X_test)
        ),
        "simple_rule": classification_metrics(
            y_test, y_pred=rule.predict(X_test), y_proba=rule.predict_proba(X_test)
        ),
        "split": {
            "n_train": len(train_df),
            "n_test": len(test_df),
            "train_churn_rate": float(y_train.mean()),
            "test_churn_rate": float(y_test.mean()),
            "random_seed": cfg["data"]["random_seed"],
            "test_size": cfg["data"]["test_size"],
        },
    }

    out_path = root / cfg["paths"]["reports_dir"] / "baselines.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)

    print("Baseline comparison:")
    for name in ("majority_class", "simple_rule"):
        row = metrics_to_row(name, results[name])
        print(
            f"  {name}: acc={row['accuracy']:.3f} "
            f"prec={row['precision']:.3f} rec={row['recall']:.3f} f1={row['f1']:.3f}"
        )
    print(f"Wrote {out_path}")
    return results


if __name__ == "__main__":
    run_baselines()

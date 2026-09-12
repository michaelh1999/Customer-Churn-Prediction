"""Helpers for comparing and explaining classification metrics."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd

from src.data.load import load_config, project_root
from src.evaluation.metrics import metrics_to_row


METRIC_GUIDE = {
    "accuracy": {
        "rewards": "Overall correct predictions (both classes).",
        "misleading_when": "Classes are imbalanced — majority guessing looks strong.",
    },
    "precision": {
        "rewards": "Fraction of predicted churners who actually churn.",
        "misleading_when": "You care about finding all churners (may ignore missed churn).",
    },
    "recall": {
        "rewards": "Fraction of actual churners correctly flagged.",
        "misleading_when": "Predicting almost everyone as churn inflates recall.",
    },
    "f1": {
        "rewards": "Harmonic balance of precision and recall.",
        "misleading_when": "Business costs of FP vs FN are asymmetric (F1 treats them equally).",
    },
    "log_loss": {
        "rewards": "Well-calibrated, confident correct probabilities (lower is better).",
        "misleading_when": "Compared across datasets with different base rates without context.",
    },
    "roc_auc": {
        "rewards": "Ranking quality: ability to score churners higher than non-churners.",
        "misleading_when": "Imbalanced data — can look strong while precision at useful recall is poor.",
    },
    "pr_auc": {
        "rewards": "Precision–recall tradeoff focused on the positive class.",
        "misleading_when": "Harder to interpret absolute values; depends on prevalence.",
    },
}


def load_metric_reports() -> pd.DataFrame:
    """Load baseline + model metric JSON files into a comparison table."""
    root = project_root()
    cfg = load_config()
    reports = root / cfg["paths"]["reports_dir"]

    rows: list[dict[str, Any]] = []

    baselines_path = reports / "baselines.json"
    if baselines_path.exists():
        with open(baselines_path, encoding="utf-8") as f:
            baselines = json.load(f)
        for name in ("majority_class", "simple_rule"):
            if name in baselines:
                rows.append(metrics_to_row(name, baselines[name]))

    for path in sorted(reports.glob("*_metrics.json")):
        with open(path, encoding="utf-8") as f:
            payload = json.load(f)
        name = payload.get("model_name", path.stem.replace("_metrics", ""))
        rows.append(metrics_to_row(name, payload.get("test_metrics", payload)))

    return pd.DataFrame(rows)


def write_metric_comparison(out_path: Path | None = None) -> pd.DataFrame:
    """Persist a CSV comparison table under reports/."""
    root = project_root()
    cfg = load_config()
    df = load_metric_reports()
    path = out_path or (root / cfg["paths"]["reports_dir"] / "metric_comparison.csv")
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)
    return df

"""Classification threshold sweeps and plots."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix

from src.data.load import load_config, project_root


def threshold_metrics(y_true, y_proba, threshold: float) -> dict[str, float | int]:
    """Compute confusion counts and rates at a single threshold."""
    y_true = np.asarray(y_true).astype(int)
    y_proba = np.asarray(y_proba, dtype=float)
    y_pred = (y_proba >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) else 0.0
    accuracy = (tp + tn) / max(len(y_true), 1)
    positive_rate = float(y_pred.mean())
    return {
        "threshold": float(threshold),
        "tp": int(tp),
        "fp": int(fp),
        "tn": int(tn),
        "fn": int(fn),
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
        "accuracy": float(accuracy),
        "positive_prediction_rate": positive_rate,
    }


def sweep_thresholds(
    y_true,
    y_proba,
    min_t: float | None = None,
    max_t: float | None = None,
    step: float | None = None,
) -> pd.DataFrame:
    """Evaluate metrics for thresholds from min to max inclusive."""
    cfg = load_config()
    tcfg = cfg.get("thresholds", {})
    min_t = min_t if min_t is not None else tcfg.get("min", 0.05)
    max_t = max_t if max_t is not None else tcfg.get("max", 0.95)
    step = step if step is not None else tcfg.get("step", 0.05)
    thresholds = np.round(np.arange(min_t, max_t + 1e-9, step), 4)
    rows = [threshold_metrics(y_true, y_proba, t) for t in thresholds]
    return pd.DataFrame(rows)


def best_threshold(sweep_df: pd.DataFrame, metric: str = "f1") -> float:
    """Return the threshold that maximizes ``metric``."""
    idx = sweep_df[metric].idxmax()
    return float(sweep_df.loc[idx, "threshold"])


def plot_threshold_curves(sweep_df: pd.DataFrame, out_path: Path | None = None) -> Path:
    """Plot precision/recall/F1/accuracy vs threshold and save PNG."""
    root = project_root()
    cfg = load_config()
    path = out_path or (root / cfg["paths"]["figures_dir"] / "threshold_metrics.png")
    path.parent.mkdir(parents=True, exist_ok=True)

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    for col in ("precision", "recall", "f1", "accuracy"):
        axes[0].plot(sweep_df["threshold"], sweep_df[col], label=col, marker="o", ms=3)
    axes[0].axvline(0.5, color="gray", ls="--", label="0.5 default")
    axes[0].set_xlabel("Threshold")
    axes[0].set_ylabel("Score")
    axes[0].set_title("Metrics vs classification threshold")
    axes[0].legend()
    axes[0].set_ylim(0, 1.05)

    axes[1].plot(
        sweep_df["threshold"],
        sweep_df["positive_prediction_rate"],
        color="#E45756",
        marker="o",
        ms=3,
        label="positive prediction rate",
    )
    axes[1].axvline(0.5, color="gray", ls="--", label="0.5 default")
    axes[1].set_xlabel("Threshold")
    axes[1].set_ylabel("Rate")
    axes[1].set_title("Share predicted as churn")
    axes[1].legend()
    axes[1].set_ylim(0, 1.05)

    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)
    return path


def run_threshold_analysis(
    y_true=None,
    y_proba=None,
    model_name: str = "logistic_regression",
) -> pd.DataFrame:
    """Load saved test probabilities if needed, sweep, plot, and save CSV."""
    root = project_root()
    cfg = load_config()
    if y_true is None or y_proba is None:
        proba_path = root / cfg["paths"]["processed_dir"] / f"{model_name}_test_proba.csv"
        if not proba_path.exists():
            raise FileNotFoundError(f"Missing {proba_path}; train the model first.")
        proba_df = pd.read_csv(proba_path)
        y_true = proba_df["y_true"]
        y_proba = proba_df["y_proba"]

    sweep = sweep_thresholds(y_true, y_proba)
    csv_path = root / cfg["paths"]["reports_dir"] / f"{model_name}_threshold_sweep.csv"
    sweep.to_csv(csv_path, index=False)
    fig_path = plot_threshold_curves(
        sweep, root / cfg["paths"]["figures_dir"] / f"{model_name}_threshold_metrics.png"
    )
    print(f"F1-optimal threshold: {best_threshold(sweep, 'f1'):.2f}")
    print(f"Wrote {csv_path}")
    print(f"Wrote {fig_path}")
    return sweep

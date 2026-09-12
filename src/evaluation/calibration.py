"""Probability calibration diagnostics."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.calibration import calibration_curve
from sklearn.metrics import brier_score_loss, log_loss

from src.data.load import load_config, project_root


def calibration_stats(y_true, y_proba, n_bins: int = 10) -> dict:
    """Return Brier score, log loss, and reliability curve points."""
    y_true = np.asarray(y_true).astype(int)
    y_proba = np.asarray(y_proba, dtype=float)
    proba_clip = np.clip(y_proba, 1e-7, 1 - 1e-7)
    frac_pos, mean_pred = calibration_curve(y_true, y_proba, n_bins=n_bins, strategy="quantile")
    return {
        "brier_score": float(brier_score_loss(y_true, y_proba)),
        "log_loss": float(log_loss(y_true, proba_clip)),
        "fraction_of_positives": frac_pos.tolist(),
        "mean_predicted_value": mean_pred.tolist(),
    }


def plot_calibration_curve(y_true, y_proba, model_name: str, n_bins: int = 10) -> Path:
    """Save a reliability diagram under reports/figures/."""
    root = project_root()
    cfg = load_config()
    stats = calibration_stats(y_true, y_proba, n_bins=n_bins)

    fig, ax = plt.subplots(figsize=(5, 5))
    ax.plot([0, 1], [0, 1], ls="--", color="gray", label="Perfectly calibrated")
    ax.plot(
        stats["mean_predicted_value"],
        stats["fraction_of_positives"],
        marker="o",
        label=model_name,
    )
    ax.set_xlabel("Mean predicted probability")
    ax.set_ylabel("Fraction of positives")
    ax.set_title(
        f"Calibration — {model_name}\n"
        f"Brier={stats['brier_score']:.3f}, log_loss={stats['log_loss']:.3f}"
    )
    ax.legend(loc="upper left")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    fig.tight_layout()

    out = root / cfg["paths"]["figures_dir"] / f"{model_name}_calibration.png"
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=120)
    plt.close(fig)
    return out


def run_calibration(model_name: str = "logistic_regression") -> dict:
    """Load saved test probabilities and write calibration figure + JSON."""
    root = project_root()
    cfg = load_config()
    proba_path = root / cfg["paths"]["processed_dir"] / f"{model_name}_test_proba.csv"
    df = pd.read_csv(proba_path)
    stats = calibration_stats(df["y_true"], df["y_proba"])
    fig_path = plot_calibration_curve(df["y_true"], df["y_proba"], model_name)
    out = {
        "model_name": model_name,
        **{k: stats[k] for k in ("brier_score", "log_loss")},
        "figure": str(fig_path.relative_to(root)),
    }
    json_path = root / cfg["paths"]["reports_dir"] / f"{model_name}_calibration.json"
    import json

    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2)
    print(out)
    return out

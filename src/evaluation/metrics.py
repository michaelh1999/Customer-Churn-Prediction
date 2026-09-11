"""Classification metrics helpers."""

from __future__ import annotations

from typing import Any

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    brier_score_loss,
    confusion_matrix,
    f1_score,
    log_loss,
    precision_score,
    recall_score,
    roc_auc_score,
)


def classification_metrics(
    y_true,
    y_pred=None,
    y_proba=None,
    threshold: float = 0.5,
) -> dict[str, Any]:
    """Compute common binary classification metrics.

    Provide either ``y_pred`` or ``y_proba`` (probabilities for the positive class).
    When only probabilities are given, predictions use ``threshold``.
    """
    y_true = np.asarray(y_true).astype(int)
    if y_proba is not None:
        y_proba = np.asarray(y_proba, dtype=float)
        if y_proba.ndim == 2:
            y_proba = y_proba[:, 1]
        if y_pred is None:
            y_pred = (y_proba >= threshold).astype(int)
    if y_pred is None:
        raise ValueError("Provide y_pred and/or y_proba.")
    y_pred = np.asarray(y_pred).astype(int)

    metrics: dict[str, Any] = {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
    }

    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    tn, fp, fn, tp = cm.ravel()
    metrics["confusion_matrix"] = {
        "tn": int(tn),
        "fp": int(fp),
        "fn": int(fn),
        "tp": int(tp),
    }

    if y_proba is not None:
        # Clip for numerical stability in log_loss
        proba_clip = np.clip(y_proba, 1e-7, 1 - 1e-7)
        metrics["log_loss"] = float(log_loss(y_true, proba_clip))
        metrics["roc_auc"] = float(roc_auc_score(y_true, y_proba))
        metrics["pr_auc"] = float(average_precision_score(y_true, y_proba))
        metrics["brier_score"] = float(brier_score_loss(y_true, y_proba))

    return metrics


def metrics_to_row(name: str, metrics: dict[str, Any]) -> dict[str, Any]:
    """Flatten metrics dict for tabular comparison."""
    row = {"model": name}
    for key in (
        "accuracy",
        "precision",
        "recall",
        "f1",
        "log_loss",
        "roc_auc",
        "pr_auc",
        "brier_score",
    ):
        if key in metrics:
            row[key] = metrics[key]
    return row

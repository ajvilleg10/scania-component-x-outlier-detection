from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import (
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)


def _valid_label_mask(y_true: np.ndarray) -> np.ndarray:
    return np.asarray(y_true) >= 0


def binary_classification_metrics(y_true, y_pred, scores=None) -> dict:
    """Compute binary metrics when labels or references are available.

    Rows with label -1 are treated as unlabeled and ignored in supervised metrics.
    """
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)

    valid = _valid_label_mask(y_true)
    y_true = y_true[valid]
    y_pred = y_pred[valid]
    scores = np.asarray(scores)[valid] if scores is not None else None

    metrics = {
        "n_samples": int(len(y_true)),
        "n_outliers": int(np.sum(y_true == 1)) if len(y_true) else 0,
        "n_normals": int(np.sum(y_true == 0)) if len(y_true) else 0,
        "n_predicted_outliers": int(np.sum(y_pred == 1)) if len(y_pred) else 0,
        "precision": precision_score(y_true, y_pred, zero_division=0) if len(y_true) else None,
        "recall": recall_score(y_true, y_pred, zero_division=0) if len(y_true) else None,
        "f1_score": f1_score(y_true, y_pred, zero_division=0) if len(y_true) else None,
    }

    if scores is not None and len(y_true) and len(np.unique(y_true)) > 1:
        metrics["roc_auc"] = roc_auc_score(y_true, scores)
        metrics["pr_auc"] = average_precision_score(y_true, scores)
    else:
        metrics["roc_auc"] = None
        metrics["pr_auc"] = None

    if len(y_true) and len(np.unique(y_true)) > 1:
        tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
        metrics.update({"tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp)})
    else:
        metrics.update({"tn": None, "fp": None, "fn": None, "tp": None})

    return metrics


def metrics_to_dataframe(model_name: str, metrics: dict, split: str = "test", level: str = "vehicle") -> pd.DataFrame:
    return pd.DataFrame([{"model": model_name, "split": split, "level": level, **metrics}])


def compare_metric_files(metric_files: list[str]) -> pd.DataFrame:
    """Load JSON metric files and return a comparison table."""
    import json
    from pathlib import Path

    rows = []
    for file in metric_files:
        with Path(file).open("r", encoding="utf-8") as f:
            rows.append(json.load(f))
    return pd.DataFrame(rows)

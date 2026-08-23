from __future__ import annotations

import numpy as np
import torch
from sklearn.metrics import precision_recall_curve
from torch.utils.data import DataLoader


@torch.no_grad()
def reconstruction_errors(model, loader: DataLoader, device: str = "auto") -> np.ndarray:
    """Compute mean squared reconstruction error per window."""
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"

    model = model.to(device)
    model.eval()

    errors = []
    for x, _ in loader:
        x = x.to(device)
        reconstructed = model(x)
        batch_errors = torch.mean((x - reconstructed) ** 2, dim=(1, 2))
        errors.extend(batch_errors.detach().cpu().numpy())

    return np.asarray(errors, dtype=float)


def percentile_threshold(scores: np.ndarray, percentile: float = 95.0) -> float:
    """Calculate an outlier threshold from validation scores."""
    scores = np.asarray(scores, dtype=float)
    if len(scores) == 0:
        raise ValueError("Cannot calculate threshold from an empty score array")
    return float(np.percentile(scores, percentile))


def best_f1_threshold(y_true: np.ndarray, scores: np.ndarray, n_grid: int | None = None) -> float:
    """Select a validation threshold that maximizes F1 when labels are available.

    Uses ``sklearn.metrics.precision_recall_curve``, which evaluates every
    distinct score value as a candidate threshold, instead of a fixed-size
    linear grid. With severe class imbalance (as in this dataset, ~2-3%
    positive vehicles) a coarse ``linspace`` grid can straddle the few
    positive scores and miss the true optimum, or land on a degenerate
    threshold that classifies almost everything as one class. ``n_grid`` is
    kept only for backward-compatible call sites and is otherwise unused.
    """
    y_true = np.asarray(y_true)
    scores = np.asarray(scores, dtype=float)
    valid = y_true >= 0
    y_true = y_true[valid]
    scores = scores[valid]

    if len(scores) == 0:
        raise ValueError("Cannot select threshold with no valid validation rows")
    if len(np.unique(y_true)) < 2:
        return percentile_threshold(scores, 95)

    precision, recall, thresholds = precision_recall_curve(y_true, scores)
    if len(thresholds) == 0:
        return percentile_threshold(scores, 95)

    denom = precision[:-1] + recall[:-1]
    f1_values = np.where(denom > 0, 2 * precision[:-1] * recall[:-1] / np.where(denom > 0, denom, 1.0), 0.0)
    if not np.any(f1_values > 0):
        return percentile_threshold(scores, 95)
    return float(thresholds[int(np.argmax(f1_values))])


def select_threshold(
    scores: np.ndarray,
    y_true: np.ndarray | None = None,
    strategy: str = "validation_percentile",
    percentile: float = 95.0,
) -> float:
    """Select threshold using validation scores only."""
    if strategy == "validation_percentile":
        return percentile_threshold(scores, percentile)
    if strategy == "best_f1_on_validation":
        if y_true is None:
            raise ValueError("y_true is required for best_f1_on_validation")
        return best_f1_threshold(y_true, scores)
    raise ValueError(f"Unknown threshold strategy: {strategy}")


def classify_outliers(scores, threshold: float) -> np.ndarray:
    """Return binary labels: 1 = outlier, 0 = inlier."""
    return (np.asarray(scores, dtype=float) > threshold).astype(int)


def threshold_rows(rows: list[dict], threshold: float, score_col: str = "outlier_score", pred_col: str = "is_outlier") -> list[dict]:
    out = []
    for row in rows:
        item = dict(row)
        item[pred_col] = int(float(item[score_col]) > threshold)
        out.append(item)
    return out

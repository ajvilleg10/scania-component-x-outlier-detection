from __future__ import annotations

import numpy as np
import torch
from sklearn.metrics import f1_score
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


def best_f1_threshold(y_true: np.ndarray, scores: np.ndarray, n_grid: int = 200) -> float:
    """Select a validation threshold that maximizes F1 when labels are available."""
    y_true = np.asarray(y_true)
    scores = np.asarray(scores, dtype=float)
    valid = y_true >= 0
    y_true = y_true[valid]
    scores = scores[valid]

    if len(scores) == 0:
        raise ValueError("Cannot select threshold with no valid validation rows")
    if len(np.unique(y_true)) < 2:
        return percentile_threshold(scores, 95)

    thresholds = np.linspace(scores.min(), scores.max(), n_grid)
    f1_values = [f1_score(y_true, (scores > t).astype(int), zero_division=0) for t in thresholds]
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


def classify_outliers(scores: np.ndarray, threshold: float) -> np.ndarray:
    """Return binary labels: 1 = outlier, 0 = inlier."""
    return (np.asarray(scores, dtype=float) > threshold).astype(int)

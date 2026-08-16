from __future__ import annotations

from statistics import mean
from typing import Any

import numpy as np


def summarize_outliers(scores, predictions) -> dict[str, Any]:
    """Return compact score/prediction summary without Pandas."""
    scores = np.asarray(scores, dtype=float)
    predictions = np.asarray(predictions, dtype=int)
    return {
        "n_windows": int(len(scores)),
        "min_score": float(np.min(scores)) if len(scores) else None,
        "mean_score": float(np.mean(scores)) if len(scores) else None,
        "median_score": float(np.median(scores)) if len(scores) else None,
        "max_score": float(np.max(scores)) if len(scores) else None,
        "outlier_window_ratio": float(np.mean(predictions)) if len(predictions) else None,
    }


def attach_predictions(vehicle_ids, y_true, scores, predictions, start_time=None, end_time=None) -> list[dict[str, Any]]:
    n = len(scores)
    start_time = [None] * n if start_time is None else list(start_time)
    end_time = [None] * n if end_time is None else list(end_time)
    return [
        {
            "vehicle_id": list(vehicle_ids)[i],
            "y_true": int(list(y_true)[i]),
            "outlier_score": float(list(scores)[i]),
            "is_outlier": int(list(predictions)[i]),
            "start_time_step": start_time[i],
            "end_time_step": end_time[i],
        }
        for i in range(n)
    ]


def aggregate_vehicle_scores(window_predictions: list[dict[str, Any]]) -> list[dict[str, Any]]:
    from scania_outliers.vehicle_level import aggregate_vehicle_scores as _agg

    return _agg(window_predictions)

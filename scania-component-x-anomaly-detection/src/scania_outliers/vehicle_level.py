from __future__ import annotations

from collections import defaultdict
from statistics import mean, median
from typing import Any, Iterable

import numpy as np


def make_window_predictions(
    vehicle_ids,
    scores,
    y_true=None,
    predictions=None,
    start_time=None,
    end_time=None,
    score_col: str = "outlier_score",
    pred_col: str = "is_outlier",
) -> list[dict[str, Any]]:
    """Create window-level prediction rows without Pandas."""
    n = len(scores)
    y_true = [-1] * n if y_true is None else list(y_true)
    predictions = [None] * n if predictions is None else list(predictions)
    start_time = [None] * n if start_time is None else list(start_time)
    end_time = [None] * n if end_time is None else list(end_time)

    rows = []
    for i in range(n):
        row = {
            "window_index": i,
            "vehicle_id": list(vehicle_ids)[i],
            score_col: float(list(scores)[i]),
            "y_true": int(y_true[i]) if y_true[i] is not None else -1,
            "start_time_step": start_time[i],
            "end_time_step": end_time[i],
        }
        if predictions[i] is not None:
            row[pred_col] = int(predictions[i])
        rows.append(row)
    return rows


def add_window_predictions(rows: Iterable[dict[str, Any]], threshold: float, score_col: str = "outlier_score", pred_col: str = "is_outlier") -> list[dict[str, Any]]:
    out = []
    for row in rows:
        item = dict(row)
        item[pred_col] = int(float(item[score_col]) > threshold)
        out.append(item)
    return out


def aggregate_vehicle_scores(
    window_rows: Iterable[dict[str, Any]],
    vehicle_col: str = "vehicle_id",
    score_col: str = "outlier_score",
    label_col: str = "y_true",
    pred_col: str | None = "is_outlier",
) -> list[dict[str, Any]]:
    """Aggregate window-level scores to vehicle/trajectory level without Pandas."""
    grouped: dict[Any, list[dict[str, Any]]] = defaultdict(list)
    for row in window_rows:
        grouped[row[vehicle_col]].append(row)

    output = []
    for vehicle_id, rows in grouped.items():
        scores = [float(r[score_col]) for r in rows]
        labels = [int(r.get(label_col, -1)) for r in rows if int(r.get(label_col, -1)) >= 0]
        preds = [int(r.get(pred_col, 0)) for r in rows if pred_col and pred_col in r]
        item = {
            vehicle_col: vehicle_id,
            "max_score": float(max(scores)),
            "mean_score": float(mean(scores)),
            "median_score": float(median(scores)),
            "p95_score": float(np.percentile(scores, 95)),
            "n_windows": int(len(scores)),
            label_col: int(max(labels)) if labels else -1,
        }
        if pred_col:
            item["any_outlier"] = int(max(preds)) if preds else 0
            item["outlier_window_ratio"] = float(mean(preds)) if preds else 0.0
        output.append(item)
    return output


def classify_vehicle_scores(
    vehicle_rows: Iterable[dict[str, Any]],
    threshold: float,
    score_col: str = "max_score",
    pred_col: str = "is_outlier",
) -> list[dict[str, Any]]:
    """Classify aggregated vehicle scores using a fixed validation threshold."""
    output = []
    for row in vehicle_rows:
        item = dict(row)
        item[pred_col] = int(float(item[score_col]) > threshold)
        output.append(item)
    return output

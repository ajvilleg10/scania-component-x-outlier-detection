from __future__ import annotations

import numpy as np
import pandas as pd


def make_window_predictions(
    vehicle_ids,
    scores,
    y_true=None,
    predictions=None,
    start_time=None,
    end_time=None,
    score_col: str = "outlier_score",
    pred_col: str = "is_outlier",
) -> pd.DataFrame:
    """Create a window-level prediction table with optional labels and metadata."""
    n = len(scores)
    df = pd.DataFrame({
        "vehicle_id": np.asarray(vehicle_ids),
        score_col: np.asarray(scores, dtype=float),
    })
    if y_true is not None:
        df["y_true"] = np.asarray(y_true, dtype=int)
    else:
        df["y_true"] = -1
    if predictions is not None:
        df[pred_col] = np.asarray(predictions, dtype=int)
    if start_time is not None:
        df["start_time"] = np.asarray(start_time)
    if end_time is not None:
        df["end_time"] = np.asarray(end_time)
    df["window_index"] = np.arange(n)
    return df


def aggregate_vehicle_scores(
    window_df: pd.DataFrame,
    vehicle_col: str = "vehicle_id",
    score_col: str = "outlier_score",
    label_col: str = "y_true",
    pred_col: str | None = "is_outlier",
) -> pd.DataFrame:
    """Aggregate window-level scores to vehicle/trajectory level.

    This is the preferred evaluation level when labels such as `class_label` are
    available only at vehicle level.
    """
    if vehicle_col not in window_df.columns:
        raise ValueError(f"Missing vehicle column: {vehicle_col}")
    if score_col not in window_df.columns:
        raise ValueError(f"Missing score column: {score_col}")

    grouped = window_df.groupby(vehicle_col, dropna=False)
    out = grouped[score_col].agg(
        max_score="max",
        mean_score="mean",
        median_score="median",
        p95_score=lambda s: float(np.percentile(s, 95)),
        n_windows="size",
    ).reset_index()

    if label_col in window_df.columns:
        # If window labels are vehicle-level labels repeated on every row, max keeps
        # one binary reference per vehicle and ignores unlabeled -1 when all unknown.
        label_summary = grouped[label_col].agg(lambda s: int(np.max(s))).reset_index(name=label_col)
        out = out.merge(label_summary, on=vehicle_col, how="left")
    else:
        out[label_col] = -1

    if pred_col and pred_col in window_df.columns:
        pred_summary = grouped[pred_col].agg(
            any_outlier="max",
            outlier_window_ratio="mean",
        ).reset_index()
        out = out.merge(pred_summary, on=vehicle_col, how="left")

    return out


def classify_vehicle_scores(
    vehicle_df: pd.DataFrame,
    threshold: float,
    score_col: str = "max_score",
    pred_col: str = "is_outlier",
) -> pd.DataFrame:
    """Classify aggregated vehicle scores using a fixed validation threshold."""
    result = vehicle_df.copy()
    result[pred_col] = (result[score_col].astype(float) > threshold).astype(int)
    return result

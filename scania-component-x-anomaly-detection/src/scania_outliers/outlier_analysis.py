from __future__ import annotations

import pandas as pd


def summarize_outliers(scores, predictions) -> pd.DataFrame:
    """Summarize outlier and inlier counts."""
    df = pd.DataFrame({"anomaly_score": scores, "is_outlier": predictions})
    summary = (
        df["is_outlier"]
        .value_counts()
        .rename_axis("is_outlier")
        .reset_index(name="count")
    )
    summary["class_name"] = summary["is_outlier"].map({0: "inlier", 1: "outlier"})
    summary["percentage"] = summary["count"] / summary["count"].sum() * 100
    return summary[["class_name", "is_outlier", "count", "percentage"]]


def attach_predictions(vehicle_ids, y_true, scores, predictions, start_time=None, end_time=None) -> pd.DataFrame:
    """Create a window-level prediction dataframe."""
    payload = {
        "vehicle_id": vehicle_ids,
        "y_true": y_true,
        "anomaly_score": scores,
        "is_outlier": predictions,
    }
    if start_time is not None:
        payload["start_time"] = start_time
    if end_time is not None:
        payload["end_time"] = end_time
    return pd.DataFrame(payload)


def aggregate_vehicle_scores(window_predictions: pd.DataFrame) -> pd.DataFrame:
    """Aggregate window scores at vehicle level for trajectory-level interpretation."""
    return (
        window_predictions
        .groupby("vehicle_id", as_index=False)
        .agg(
            y_true=("y_true", "max"),
            max_anomaly_score=("anomaly_score", "max"),
            mean_anomaly_score=("anomaly_score", "mean"),
            outlier_windows=("is_outlier", "sum"),
            total_windows=("is_outlier", "count"),
        )
        .assign(outlier_window_ratio=lambda x: x["outlier_windows"] / x["total_windows"])
    )

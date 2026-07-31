from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Literal, Optional

import numpy as np
import pandas as pd

try:
    from pyspark.sql import DataFrame
except Exception:  # pragma: no cover - helps local tests without Spark
    DataFrame = object  # type: ignore

LabelPolicy = Literal["vehicle_label", "max_label_in_window", "unlabeled"]


@dataclass
class WindowData:
    """Container for fixed-length time windows and metadata."""

    X: np.ndarray
    vehicle_ids: np.ndarray
    y: np.ndarray
    start_time: np.ndarray
    end_time: np.ndarray
    feature_cols: list[str]


class TimeWindowBuilder:
    """Build fixed-length multivariate time windows by vehicle.

    The builder supports controlled conversion to Pandas for Colab. For the full
    dataset, use vehicle subsets or batches to avoid loading everything at once.
    """

    def __init__(
        self,
        vehicle_col: str = "vehicle_id",
        time_col: str = "time_step",
        window_size: int = 30,
        stride: int = 5,
        label_policy: LabelPolicy = "vehicle_label",
    ):
        self.vehicle_col = vehicle_col
        self.time_col = time_col
        self.window_size = window_size
        self.stride = stride
        self.label_policy = label_policy

    def _window_label(self, row_labels: Optional[np.ndarray], start: int, end: int) -> int:
        if row_labels is None or self.label_policy == "unlabeled":
            return -1
        if self.label_policy == "vehicle_label":
            # vehicle-level labels are repeated on every row after joining; use the
            # first available label in the window instead of implying time labels.
            valid = row_labels[start:end][row_labels[start:end] >= 0]
            return int(valid[0]) if len(valid) else -1
        if self.label_policy == "max_label_in_window":
            return int(np.max(row_labels[start:end]))
        raise ValueError(f"Unknown label_policy: {self.label_policy}")

    def build_from_pandas(
        self,
        pdf: pd.DataFrame,
        feature_cols: list[str],
        label_col: Optional[str] = None,
    ) -> WindowData:
        windows, vehicle_ids, labels, starts, ends = [], [], [], [], []

        for vehicle_id, group in pdf.groupby(self.vehicle_col, sort=False):
            group = group.sort_values(self.time_col)
            values = group[feature_cols].to_numpy(dtype=np.float32)
            times = group[self.time_col].to_numpy()

            row_labels = None
            if label_col and label_col in group.columns:
                row_labels = group[label_col].to_numpy(dtype=np.int64)

            for start in range(0, max(len(values) - self.window_size + 1, 0), self.stride):
                end = start + self.window_size
                windows.append(values[start:end])
                vehicle_ids.append(vehicle_id)
                starts.append(times[start])
                ends.append(times[end - 1])
                labels.append(self._window_label(row_labels, start, end))

        if not windows:
            return WindowData(
                X=np.empty((0, self.window_size, len(feature_cols)), dtype=np.float32),
                vehicle_ids=np.array([], dtype=object),
                y=np.array([], dtype=np.int64),
                start_time=np.array([], dtype=object),
                end_time=np.array([], dtype=object),
                feature_cols=feature_cols,
            )

        return WindowData(
            X=np.stack(windows),
            vehicle_ids=np.array(vehicle_ids),
            y=np.array(labels, dtype=np.int64),
            start_time=np.array(starts),
            end_time=np.array(ends),
            feature_cols=feature_cols,
        )

    def build_from_spark(
        self,
        df: DataFrame,
        feature_cols: list[str],
        max_vehicles: Optional[int] = None,
        label_col: Optional[str] = None,
    ) -> WindowData:
        if max_vehicles is not None:
            vehicles = [r[self.vehicle_col] for r in df.select(self.vehicle_col).distinct().limit(max_vehicles).collect()]
            df = df.where(df[self.vehicle_col].isin(vehicles))

        cols = [self.vehicle_col, self.time_col] + feature_cols
        if label_col and label_col in df.columns:
            cols.append(label_col)

        pdf = df.select(cols).toPandas()
        return self.build_from_pandas(pdf, feature_cols, label_col=label_col)

    @staticmethod
    def save_npz(path: str | Path, window_data: WindowData) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            path,
            X=window_data.X,
            vehicle_ids=window_data.vehicle_ids,
            y=window_data.y,
            start_time=window_data.start_time,
            end_time=window_data.end_time,
            feature_cols=np.array(window_data.feature_cols, dtype=object),
        )

    @staticmethod
    def load_npz(path: str | Path) -> WindowData:
        data = np.load(path, allow_pickle=True)
        return WindowData(
            X=data["X"],
            vehicle_ids=data["vehicle_ids"],
            y=data["y"],
            start_time=data["start_time"],
            end_time=data["end_time"],
            feature_cols=list(data["feature_cols"]),
        )

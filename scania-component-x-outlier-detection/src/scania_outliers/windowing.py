from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Literal, Optional, Sequence

import numpy as np

LabelPolicy = Literal["vehicle_label", "max_label_in_window", "unlabeled"]


@dataclass
class WindowData:
    """Small in-memory container kept for unit tests and lightweight examples."""

    X: np.ndarray
    vehicle_ids: np.ndarray
    y: np.ndarray
    start_time: np.ndarray
    end_time: np.ndarray
    feature_cols: list[str]


class TimeWindowBuilder:
    """Build fixed-length windows from in-memory Python records.

    The production pipeline uses SparkWindowBuilder and Parquet outputs. This
    class remains only for small tests/examples and does not depend on Pandas.
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
            valid = row_labels[start:end][row_labels[start:end] >= 0]
            return int(valid[0]) if len(valid) else -1
        if self.label_policy == "max_label_in_window":
            return int(np.max(row_labels[start:end]))
        raise ValueError(f"Unknown label_policy: {self.label_policy}")

    def build_from_records(self, records: Sequence[dict], feature_cols: list[str], label_col: Optional[str] = None) -> WindowData:
        grouped: dict[object, list[dict]] = {}
        for row in records:
            grouped.setdefault(row[self.vehicle_col], []).append(row)

        windows, vehicle_ids, labels, starts, ends = [], [], [], [], []
        for vehicle_id, rows in grouped.items():
            rows = sorted(rows, key=lambda r: r[self.time_col])
            values = np.asarray([[row[c] for c in feature_cols] for row in rows], dtype=np.float32)
            times = np.asarray([row[self.time_col] for row in rows])
            row_labels = None
            if label_col:
                row_labels = np.asarray([row.get(label_col, -1) for row in rows], dtype=np.int64)
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

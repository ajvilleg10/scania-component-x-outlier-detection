from __future__ import annotations

from pathlib import Path
from typing import Any, Iterator

import numpy as np
import torch
from torch.utils.data import Dataset, IterableDataset


class WindowDataset(Dataset):
    """In-memory PyTorch dataset kept for tests and small synthetic examples."""

    def __init__(self, X: np.ndarray, y: np.ndarray | None = None):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.long) if y is not None else torch.full((len(X),), -1, dtype=torch.long)

    def __len__(self) -> int:
        return self.X.shape[0]

    def __getitem__(self, idx: int):
        x = self.X[idx]
        return x, x

    def labels_numpy(self) -> np.ndarray:
        return self.y.detach().cpu().numpy()


def to_tensor_dataset(window_data) -> WindowDataset:
    """Convert a small WindowData object into a PyTorch reconstruction dataset."""
    return WindowDataset(window_data.X, getattr(window_data, "y", None))


def _get_pyarrow_dataset(parquet_dir: str | Path):
    try:
        import pyarrow.dataset as ds
    except ModuleNotFoundError as exc:  # pragma: no cover
        raise ModuleNotFoundError(
            "pyarrow is required to stream Parquet windows into PyTorch. Install requirements.txt."
        ) from exc
    return ds.dataset(str(parquet_dir), format="parquet")


class ParquetWindowDataset(IterableDataset):
    """Iterable PyTorch dataset that streams Spark-generated Parquet windows.

    Each yielded item is already a batch `(X, X)` for autoencoder training, so use
    `DataLoader(dataset, batch_size=None)`. No Pandas conversion is used.
    """

    def __init__(self, parquet_dir: str | Path, batch_size: int = 512):
        self.parquet_dir = str(parquet_dir)
        self.batch_size = int(batch_size)
        self._n_rows: int | None = None

    def __len__(self) -> int:
        if self._n_rows is None:
            self._n_rows = int(_get_pyarrow_dataset(self.parquet_dir).count_rows())
        return self._n_rows

    def __iter__(self) -> Iterator[tuple[torch.Tensor, torch.Tensor]]:
        dataset = _get_pyarrow_dataset(self.parquet_dir)
        for batch in dataset.to_batches(columns=["X"], batch_size=self.batch_size):
            x_list = batch.column("X").to_pylist()
            X = torch.tensor(x_list, dtype=torch.float32)
            yield X, X


def make_parquet_loader(parquet_dir: str | Path, batch_size: int = 512):
    """Create a DataLoader over Parquet windows without Pandas."""
    from torch.utils.data import DataLoader

    return DataLoader(ParquetWindowDataset(parquet_dir, batch_size=batch_size), batch_size=None)


def inspect_parquet_windows(parquet_dir: str | Path) -> dict[str, int]:
    """Return row count, window size and feature count from a Parquet window directory."""
    dataset = _get_pyarrow_dataset(parquet_dir)
    n_rows = int(dataset.count_rows())
    window_size = 0
    n_features = 0
    for batch in dataset.to_batches(columns=["X"], batch_size=1):
        rows = batch.column("X").to_pylist()
        if rows:
            window_size = len(rows[0])
            n_features = len(rows[0][0]) if rows[0] else 0
        break
    return {"n_rows": n_rows, "window_size": window_size, "n_features": n_features}


@torch.no_grad()
def compute_feature_error_scale(
    model,
    parquet_dir: str | Path,
    batch_size: int = 512,
    device: str = "auto",
    max_windows: int | None = 20000,
    robust_percentile: float = 90.0,
) -> np.ndarray:
    """Calibrate a per-feature reconstruction-error scale from normal windows.

    Some sensor channels are intrinsically noisier or harder to reconstruct than
    others even under fully normal operation. Combining raw per-window squared
    error with a flat mean over all channels lets those channels dominate the
    outlier score and mask the contribution of smaller, but more informative,
    channels. This function estimates, per feature, a robust "typical normal
    error" scale (a high percentile of the per-window mean squared error) using
    windows drawn from the training population, which is normal-only by
    construction (see ``filter_normal_training_vehicles``). The scale is meant to
    be passed to :func:`score_parquet_windows` via ``feature_scale`` so every
    channel contributes comparably to the final score, regardless of its own
    baseline noise level.
    """
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    model = model.to(device)
    model.eval()

    dataset = _get_pyarrow_dataset(parquet_dir)
    per_feature_errors: list[np.ndarray] = []
    seen = 0
    for batch in dataset.to_batches(columns=["X"], batch_size=batch_size):
        x_list = batch.column("X").to_pylist()
        X = torch.tensor(x_list, dtype=torch.float32, device=device)
        reconstructed = model(X)
        # Mean squared error per window, per feature (averaged over the time axis only).
        err = torch.mean((X - reconstructed) ** 2, dim=1).detach().cpu().numpy()  # (batch, n_features)
        per_feature_errors.append(err)
        seen += err.shape[0]
        if max_windows is not None and seen >= max_windows:
            break

    if not per_feature_errors:
        raise ValueError(f"No windows found to calibrate feature error scale in {parquet_dir}")

    stacked = np.concatenate(per_feature_errors, axis=0)
    scale = np.percentile(stacked, robust_percentile, axis=0)
    positive = scale[scale > 0]
    # Features that are (near-)perfectly reconstructed under normal conditions get
    # a scale of ~0. Flooring them with a tiny epsilon would blow up their
    # normalized error into huge, meaningless values purely from floating-point
    # noise, dominating the median just as badly as the original problem — only
    # via a different mechanism. Falling back to the median scale across the
    # other features treats a degenerate channel as "typical", not "infinitely
    # sensitive".
    fallback = float(np.median(positive)) if positive.size else 1e-6
    scale = np.where(scale > 0, scale, fallback)
    return scale.astype(float)


@torch.no_grad()
def score_parquet_windows(
    model,
    parquet_dir: str | Path,
    batch_size: int = 512,
    device: str = "auto",
    feature_scale: np.ndarray | None = None,
) -> list[dict[str, Any]]:
    """Run reconstruction scoring over Parquet windows and return prediction rows.

    The returned rows are window-level records containing metadata and an
    `outlier_score`. They can later be classified and aggregated at vehicle level.

    When ``feature_scale`` is provided (see :func:`compute_feature_error_scale`),
    each feature's squared error is first normalized by its own calibrated scale
    — so every channel contributes on a comparable footing regardless of its
    raw units or inherent noise level — and the window score is the mean
    across normalized channels. Normalizing first (rather than taking a robust
    aggregate like the median afterwards) matters here: a median would treat a
    real degradation signal that only shows up in a minority of channels as
    noise and dilute it away. When ``feature_scale`` is ``None`` the function
    falls back to the original flat mean over time and features, kept for
    backward compatibility.
    """
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"

    model = model.to(device)
    model.eval()

    dataset = _get_pyarrow_dataset(parquet_dir)
    rows: list[dict[str, Any]] = []
    columns = ["window_id", "vehicle_id", "start_time_step", "end_time_step", "y_true", "X"]

    scale_tensor = None
    if feature_scale is not None:
        scale_tensor = torch.tensor(feature_scale, dtype=torch.float32, device=device)

    for batch in dataset.to_batches(columns=columns, batch_size=batch_size):
        x_list = batch.column("X").to_pylist()
        X = torch.tensor(x_list, dtype=torch.float32, device=device)
        reconstructed = model(X)

        if scale_tensor is None:
            scores = torch.mean((X - reconstructed) ** 2, dim=(1, 2)).detach().cpu().numpy()
        else:
            per_feature = torch.mean((X - reconstructed) ** 2, dim=1)  # (batch, n_features)
            normalized = per_feature / scale_tensor.unsqueeze(0)
            scores = torch.mean(normalized, dim=1).detach().cpu().numpy()

        window_ids = batch.column("window_id").to_pylist()
        vehicle_ids = batch.column("vehicle_id").to_pylist()
        starts = batch.column("start_time_step").to_pylist()
        ends = batch.column("end_time_step").to_pylist()
        labels = batch.column("y_true").to_pylist()

        for i, score in enumerate(scores):
            rows.append({
                "window_id": window_ids[i],
                "vehicle_id": vehicle_ids[i],
                "start_time_step": starts[i],
                "end_time_step": ends[i],
                "y_true": int(labels[i]) if labels[i] is not None else -1,
                "outlier_score": float(score),
            })
    return rows

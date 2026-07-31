from __future__ import annotations

import numpy as np
import torch
from torch.utils.data import Dataset


class WindowDataset(Dataset):
    """PyTorch dataset for multivariate time windows.

    Autoencoders use the input as target. Labels are retained for evaluation outside
    the training loop.
    """

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
    """Convert a WindowData object into a PyTorch reconstruction dataset."""
    return WindowDataset(window_data.X, getattr(window_data, "y", None))

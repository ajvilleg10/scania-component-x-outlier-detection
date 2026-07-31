from __future__ import annotations

import copy
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Optional

import torch
from torch import nn
from torch.utils.data import DataLoader
from tqdm.auto import tqdm


@dataclass
class TrainingConfig:
    epochs: int = 20
    learning_rate: float = 1e-3
    batch_size: int = 128
    patience: int = 5
    device: str = "auto"


def resolve_device(device: str = "auto") -> torch.device:
    if device == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(device)


class AutoencoderTrainer:
    """Generic trainer for reconstruction-based anomaly detection models."""

    def __init__(self, model: nn.Module, config: TrainingConfig):
        self.model = model
        self.config = config
        self.device = resolve_device(config.device)
        self.model.to(self.device)

    def fit(self, train_loader: DataLoader, val_loader: Optional[DataLoader] = None) -> Dict[str, list | float]:
        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.config.learning_rate)
        criterion = nn.MSELoss()
        history: Dict[str, list | float] = {"train_loss": [], "val_loss": [], "training_time_seconds": 0.0}

        best_val = float("inf")
        best_state = copy.deepcopy(self.model.state_dict())
        patience_counter = 0
        start_time = time.time()

        for epoch in range(1, self.config.epochs + 1):
            self.model.train()
            train_loss = 0.0

            for x, target in tqdm(train_loader, desc=f"Epoch {epoch}", leave=False):
                x = x.to(self.device)
                target = target.to(self.device)

                optimizer.zero_grad()
                reconstructed = self.model(x)
                loss = criterion(reconstructed, target)
                loss.backward()
                optimizer.step()

                train_loss += loss.item() * x.size(0)

            train_loss /= max(len(train_loader.dataset), 1)
            history["train_loss"].append(train_loss)  # type: ignore[index]

            if val_loader is not None:
                val_loss = self.evaluate_loss(val_loader)
                history["val_loss"].append(val_loss)  # type: ignore[index]

                if val_loss < best_val:
                    best_val = val_loss
                    best_state = copy.deepcopy(self.model.state_dict())
                    patience_counter = 0
                else:
                    patience_counter += 1

                if patience_counter >= self.config.patience:
                    print(f"Early stopping at epoch {epoch}")
                    break

            message = f"Epoch {epoch:03d} | train_loss={train_loss:.6f}"
            if val_loader is not None:
                message += f" | val_loss={history['val_loss'][-1]:.6f}"  # type: ignore[index]
            print(message)

        history["training_time_seconds"] = round(time.time() - start_time, 3)
        self.model.load_state_dict(best_state)
        return history

    @torch.no_grad()
    def evaluate_loss(self, loader: DataLoader) -> float:
        self.model.eval()
        criterion = nn.MSELoss(reduction="sum")
        total = 0.0
        n = 0

        for x, target in loader:
            x = x.to(self.device)
            target = target.to(self.device)
            reconstructed = self.model(x)
            total += criterion(reconstructed, target).item()
            n += x.numel()

        return total / max(n, 1)

    def save(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(self.model.state_dict(), path)

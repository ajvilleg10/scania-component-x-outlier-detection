from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
from sklearn.metrics import PrecisionRecallDisplay, RocCurveDisplay


def save_missing_values_plot(missing_report: pd.DataFrame, output_path: str | Path, top_n: int = 30) -> None:
    """Save a bar chart with top missing-value columns."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    data = missing_report.head(top_n).sort_values("missing_ratio")
    plt.figure(figsize=(10, max(5, len(data) * 0.25)))
    plt.barh(data["column"], data["missing_ratio"] * 100)
    plt.xlabel("Missing values (%)")
    plt.ylabel("Column")
    plt.title(f"Top {top_n} columns by missing ratio")
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()


def save_score_distribution(scores, output_path: str | Path, title: str = "Anomaly score distribution") -> None:
    """Save histogram of anomaly scores."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    plt.figure(figsize=(10, 5))
    plt.hist(scores, bins=50)
    plt.xlabel("Anomaly score")
    plt.ylabel("Frequency")
    plt.title(title)
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()


def save_training_history(history: dict, output_path: str | Path, title: str = "Training history") -> None:
    """Save train and validation loss curves."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    plt.figure(figsize=(10, 5))
    plt.plot(history.get("train_loss", []), label="train_loss")
    if history.get("val_loss"):
        plt.plot(history["val_loss"], label="val_loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title(title)
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()


def save_roc_pr_curves(y_true, scores, roc_path: str | Path, pr_path: str | Path) -> None:
    """Save ROC and Precision-Recall curves when labels are available."""
    roc_path = Path(roc_path)
    pr_path = Path(pr_path)
    roc_path.parent.mkdir(parents=True, exist_ok=True)
    pr_path.parent.mkdir(parents=True, exist_ok=True)

    RocCurveDisplay.from_predictions(y_true, scores)
    plt.tight_layout()
    plt.savefig(roc_path, dpi=300)
    plt.close()

    PrecisionRecallDisplay.from_predictions(y_true, scores)
    plt.tight_layout()
    plt.savefig(pr_path, dpi=300)
    plt.close()

from pathlib import Path

import numpy as np

from scania_outliers.visualization import (
    save_correlation_matrix,
    save_feature_boxplots,
    save_feature_distributions,
    save_training_history_plot,
    save_original_class_distribution_plot,
)


def test_basic_figures_are_created(tmp_path: Path):
    rng = np.random.default_rng(42)
    values = rng.normal(size=(100, 4))
    names = ["a", "b", "c", "d"]
    files = [
        tmp_path / "dist.png",
        tmp_path / "box.png",
        tmp_path / "corr.png",
        tmp_path / "loss.png",
        tmp_path / "classes.png",
    ]
    save_feature_distributions(values, names, files[0])
    save_feature_boxplots(values, names, files[1])
    save_correlation_matrix(values, names, files[2])
    save_training_history_plot({"train_loss": [1.0, 0.8], "val_loss": [1.1, 0.9]}, files[3], "model")
    save_original_class_distribution_plot({"validation_labels": {0: 90, 1: 3, 2: 2, 3: 2, 4: 3}}, files[4])
    assert all(p.exists() and p.stat().st_size > 0 for p in files)

"""Regression test for the feature-calibrated reconstruction-error scoring.

This test builds a small synthetic scenario that mirrors the failure mode
found in the real SCANIA run: one feature ("noisy_benign") has large
reconstruction error under fully normal conditions, unrelated to any repair
event, while a second feature ("degradation_signal") has a small but
consistent extra error only for outlier vehicles. With the original flat
mean over all features, the noisy-but-benign feature swamps the signal and
ROC-AUC collapses to ~0.5. With per-feature calibration
(`compute_feature_error_scale`) and a median aggregate
(`score_parquet_windows(..., feature_scale=...)`), the two classes should
separate cleanly (ROC-AUC close to 1.0).

This does not claim anything about the real SCANIA Component X results,
which require the actual dataset and a full Colab run to re-evaluate; it
only proves the scoring mechanism itself behaves as intended.
"""

from __future__ import annotations

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import torch
from sklearn.metrics import roc_auc_score
from torch import nn

from scania_outliers.datasets import compute_feature_error_scale, score_parquet_windows

N_TIME = 10
N_FEATURES = 3  # feature 0: noisy benign, feature 1: degradation signal, feature 2: quiet/inert


class ZeroForNoisyAndSignalFeaturesAutoencoder(nn.Module):
    """Fake 'model' whose reconstruction quality differs by feature, on purpose.

    - feature 0 ("noisy_benign"): reconstruction is always 0, so its squared
      error tracks the feature's own large, class-independent noise.
    - feature 1 ("degradation_signal"): reconstruction is also always 0 (the
      autoencoder was only ever trained on the small normal baseline and never
      learned the extra outlier shift), so its squared error is small but
      genuinely larger, on average, for outlier windows.
    - feature 2 ("quiet_inert"): reconstruction equals the input exactly
      (near-perfect), so its error is ~0 for both classes and carries no
      signal — this is the edge case that stresses the scale-fallback logic.
    """

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        reconstructed = x.clone()
        reconstructed[:, :, 0] = 0.0
        reconstructed[:, :, 1] = 0.0
        return reconstructed


def _write_windows_parquet(path, X: np.ndarray, y_true: np.ndarray):
    table = pa.table(
        {
            "window_id": list(range(len(X))),
            "vehicle_id": [f"veh_{i}" for i in range(len(X))],
            "start_time_step": [0] * len(X),
            "end_time_step": [N_TIME] * len(X),
            "y_true": y_true.tolist(),
            "X": [w.tolist() for w in X],
        }
    )
    pq.write_table(table, str(path))


def _make_windows(rng, n_normal: int, n_outlier: int):
    # Feature 0 ("noisy_benign"): large-amplitude noise, IDENTICAL for both classes.
    noisy = rng.normal(loc=0.0, scale=5.0, size=(n_normal + n_outlier, N_TIME, 1))
    # Feature 1 ("degradation_signal"): small extra offset only for outliers.
    base_signal = rng.normal(loc=0.0, scale=0.1, size=(n_normal + n_outlier, N_TIME, 1))
    extra = np.zeros((n_normal + n_outlier, N_TIME, 1))
    extra[n_normal:, :, :] = 0.3  # small but consistent shift for outliers only
    degradation = base_signal + extra
    # Feature 2 ("quiet/inert"): near-zero, uninformative, low noise.
    quiet = rng.normal(loc=0.0, scale=0.05, size=(n_normal + n_outlier, N_TIME, 1))

    X = np.concatenate([noisy, degradation, quiet], axis=2).astype(np.float32)
    y_true = np.array([0] * n_normal + [1] * n_outlier, dtype=int)
    return X, y_true


def test_flat_mean_score_fails_to_separate_when_one_feature_is_noisy(tmp_path):
    rng = np.random.default_rng(7)
    X_train, _ = _make_windows(rng, n_normal=400, n_outlier=0)
    X_eval, y_eval = _make_windows(rng, n_normal=150, n_outlier=150)

    train_dir = tmp_path / "train"
    eval_dir = tmp_path / "eval"
    train_dir.mkdir()
    eval_dir.mkdir()
    _write_windows_parquet(train_dir / "part.parquet", X_train, np.zeros(len(X_train), dtype=int))
    _write_windows_parquet(eval_dir / "part.parquet", X_eval, y_eval)

    # The model reconstructs feature 0 (noisy_benign) and feature 1
    # (degradation_signal) as flat zero, and feature 2 (quiet_inert) almost
    # perfectly. Only feature 1's error genuinely differs between classes.
    model = ZeroForNoisyAndSignalFeaturesAutoencoder()

    # Original behavior: flat mean over time AND features (feature_scale=None).
    rows_flat = score_parquet_windows(model, eval_dir, batch_size=64, device="cpu", feature_scale=None)
    scores_flat = np.array([r["outlier_score"] for r in rows_flat])
    labels_flat = np.array([r["y_true"] for r in rows_flat])
    auc_flat = roc_auc_score(labels_flat, scores_flat)

    # Calibrated behavior: per-feature scale fitted on normal (train) windows,
    # then median aggregate across features.
    feature_scale = compute_feature_error_scale(model, train_dir, batch_size=64, device="cpu")
    rows_calibrated = score_parquet_windows(
        model, eval_dir, batch_size=64, device="cpu", feature_scale=feature_scale
    )
    scores_calibrated = np.array([r["outlier_score"] for r in rows_calibrated])
    labels_calibrated = np.array([r["y_true"] for r in rows_calibrated])
    auc_calibrated = roc_auc_score(labels_calibrated, scores_calibrated)

    # The noisy-but-benign feature should dominate the flat-mean score enough
    # that it is no better than chance at separating the classes.
    assert auc_flat < 0.65, f"expected the flat-mean score to be near chance, got AUC={auc_flat:.3f}"
    # The calibrated score should recover a strong separation.
    assert auc_calibrated > 0.9, f"expected the calibrated score to separate classes well, got AUC={auc_calibrated:.3f}"

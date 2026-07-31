"""Backward-compatible wrapper around outlier-detection utilities.

Earlier notebooks imported `scania_anomaly.anomaly_detection`. The final TFM uses
outlier terminology, but these aliases avoid breaking previous prototypes.
"""

from scania_anomaly.outlier_detection import (  # noqa: F401
    best_f1_threshold,
    classify_outliers,
    percentile_threshold,
    reconstruction_errors,
    select_threshold,
)

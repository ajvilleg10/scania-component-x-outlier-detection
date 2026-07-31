# Backward compatibility tests for earlier imports.
import numpy as np

from scania_anomaly.anomaly_detection import classify_outliers, percentile_threshold, select_threshold


def test_legacy_anomaly_module_aliases_outlier_functions():
    scores = np.array([0.1, 0.2, 0.3, 10.0])
    threshold = percentile_threshold(scores, 75)
    preds = classify_outliers(scores, threshold)
    assert preds.tolist() == [0, 0, 0, 1]


def test_legacy_select_threshold_best_f1():
    scores = np.array([0.1, 0.2, 0.8, 0.9])
    y_true = np.array([0, 0, 1, 1])
    threshold = select_threshold(scores, y_true=y_true, strategy="best_f1_on_validation")
    assert 0.1 <= threshold <= 0.9

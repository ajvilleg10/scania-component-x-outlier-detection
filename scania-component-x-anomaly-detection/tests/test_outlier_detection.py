import numpy as np

from scania_anomaly.outlier_detection import classify_outliers, percentile_threshold, select_threshold


def test_percentile_threshold_and_classification():
    scores = np.array([0.1, 0.2, 0.3, 10.0])
    threshold = percentile_threshold(scores, 75)
    preds = classify_outliers(scores, threshold)

    assert threshold > 0.3
    assert preds.tolist() == [0, 0, 0, 1]


def test_select_threshold_best_f1():
    scores = np.array([0.1, 0.2, 0.8, 0.9])
    y_true = np.array([0, 0, 1, 1])
    threshold = select_threshold(scores, y_true=y_true, strategy="best_f1_on_validation")

    assert 0.1 <= threshold <= 0.9

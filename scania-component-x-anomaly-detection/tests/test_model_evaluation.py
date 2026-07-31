import numpy as np

from scania_anomaly.model_evaluation import binary_classification_metrics


def test_binary_metrics_ignore_unlabeled_rows():
    y_true = np.array([0, 1, 1, -1])
    y_pred = np.array([0, 1, 0, 1])
    scores = np.array([0.1, 0.9, 0.4, 0.8])

    metrics = binary_classification_metrics(y_true, y_pred, scores)

    assert metrics["n_samples"] == 3
    assert metrics["precision"] == 1.0
    assert metrics["recall"] == 0.5
    assert metrics["f1_score"] > 0
    assert metrics["pr_auc"] is not None

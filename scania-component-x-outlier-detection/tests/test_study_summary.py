import json
from pathlib import Path

from scania_outliers.study_summary import build_study_summary


def _write_metric(root: Path, run: str, model: str, pr_auc: float, f1: float) -> None:
    metrics = root / "runs" / run / "metrics"
    metrics.mkdir(parents=True, exist_ok=True)
    payload = {
        "model": model,
        "level": "vehicle",
        "split": "test",
        "precision": f1,
        "recall": f1,
        "f1_score": f1,
        "pr_auc": pr_auc,
        "roc_auc": min(0.99, pr_auc + 0.1),
        "training_time_seconds": 10.0,
        "inference_time_seconds": 2.0,
    }
    (metrics / f"{model}_test_vehicle_metrics.json").write_text(json.dumps(payload), encoding="utf-8")


def test_study_summary_consolidates_runs(tmp_path: Path):
    experiments = tmp_path / "experiments"
    _write_metric(experiments, "debug_025", "lstm_autoencoder", 0.55, 0.50)
    _write_metric(experiments, "debug_025", "transformer_encoder", 0.65, 0.60)
    _write_metric(experiments, "full", "lstm_autoencoder", 0.70, 0.68)
    _write_metric(experiments, "full", "transformer_encoder", 0.80, 0.75)

    summary = build_study_summary({"paths": {"experiments_dir": str(experiments)}})

    out = experiments / "study_summary"
    assert (out / "tables" / "metrics_across_runs.csv").exists()
    assert (out / "tables" / "best_model_by_run.csv").exists()
    assert (out / "figures" / "pr_auc_across_runs.png").exists()
    assert summary["completed_runs"] == ["debug_025", "full"]

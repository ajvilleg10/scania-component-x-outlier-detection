import csv
import json
from pathlib import Path

from scania_outliers.learning_curve_summary import build_learning_curve_summary, collect_learning_curve_points


def _write_run(root: Path, run_name: str, model: str, train_n_vehicles: int | None, pr_auc: float, f1: float) -> None:
    metrics_dir = root / "runs" / run_name / "metrics"
    metrics_dir.mkdir(parents=True, exist_ok=True)
    payload = {
        "model": model,
        "level": "vehicle",
        "split": "test",
        "precision": f1,
        "recall": f1,
        "f1_score": f1,
        "pr_auc": pr_auc,
        "roc_auc": min(0.99, pr_auc + 0.1),
    }
    (metrics_dir / f"{model}_test_vehicle_metrics.json").write_text(json.dumps(payload), encoding="utf-8")

    if train_n_vehicles is not None:
        prep_dir = root / "runs" / run_name / "tables" / "preprocessing"
        prep_dir.mkdir(parents=True, exist_ok=True)
        with (prep_dir / "window_summary_by_split.csv").open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=["split", "n_windows", "n_vehicles"])
            writer.writeheader()
            writer.writerow({"split": "train", "n_windows": train_n_vehicles * 4, "n_vehicles": train_n_vehicles})
            writer.writerow({"split": "validation", "n_windows": 500, "n_vehicles": 2898})
            writer.writerow({"split": "test", "n_windows": 500, "n_vehicles": 2883})


def test_collect_points_parses_sizes_and_seeds(tmp_path: Path):
    experiments = tmp_path / "experiments"
    _write_run(experiments, "learning_curve_n025_s1", "lstm_autoencoder", None, 0.30, 0.25)
    _write_run(experiments, "learning_curve_n025_s2", "lstm_autoencoder", None, 0.36, 0.31)
    _write_run(experiments, "learning_curve_n025_s3", "lstm_autoencoder", None, 0.33, 0.28)
    _write_run(experiments, "learning_curve_n050_s1", "lstm_autoencoder", None, 0.45, 0.40)
    _write_run(experiments, "full", "lstm_autoencoder", 15208, 0.70, 0.65)
    # A debug_* run should never be picked up by the learning-curve collector.
    _write_run(experiments, "debug_025", "lstm_autoencoder", 25, 0.10, 0.05)

    rows = collect_learning_curve_points(experiments, "lstm_autoencoder")
    run_names = {r["run_name"] for r in rows}
    assert run_names == {
        "learning_curve_n025_s1", "learning_curve_n025_s2", "learning_curve_n025_s3",
        "learning_curve_n050_s1", "full",
    }
    n25 = [r for r in rows if r["run_name"] == "learning_curve_n025_s1"][0]
    assert n25["n_vehicles"] == 25
    assert n25["seed"] == 1
    assert n25["is_anchor"] is False

    anchor = [r for r in rows if r["run_name"] == "full"][0]
    assert anchor["n_vehicles"] == 15208
    assert anchor["is_anchor"] is True


def test_build_summary_produces_table_and_figures(tmp_path: Path):
    experiments = tmp_path / "experiments"
    _write_run(experiments, "learning_curve_n025_s1", "lstm_autoencoder", None, 0.30, 0.25)
    _write_run(experiments, "learning_curve_n025_s2", "lstm_autoencoder", None, 0.36, 0.31)
    _write_run(experiments, "learning_curve_n050_s1", "lstm_autoencoder", None, 0.45, 0.40)
    _write_run(experiments, "learning_curve_n050_s2", "lstm_autoencoder", None, 0.49, 0.44)
    _write_run(experiments, "full", "lstm_autoencoder", 15208, 0.70, 0.65)

    config = {
        "paths": {"experiments_dir": str(experiments)},
        "learning_curve": {"model": "lstm_autoencoder"},
    }
    summary = build_learning_curve_summary(config)

    out = experiments / "learning_curve_summary"
    assert (out / "tables" / "learning_curve_points.csv").exists()
    assert (out / "figures" / "learning_curve_pr_auc.png").exists()
    assert (out / "figures" / "learning_curve_f1_score.png").exists()
    assert summary["sizes_with_repeats"] == 2  # 25 and 50 each have >1 seed
    assert summary["n_points"] == 5

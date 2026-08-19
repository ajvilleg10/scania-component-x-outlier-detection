from __future__ import annotations

import csv
import json
import re
from pathlib import Path
from typing import Any

from scania_outliers.visualization import save_cross_run_metric_plot


def _run_order(name: str):
    match = re.fullmatch(r"debug_(\d+)", name)
    if match:
        return (0, int(match.group(1)))
    if name == "full":
        return (1, 10**12)
    return (2, name)


def _write_csv(rows: list[dict[str, Any]], path: Path) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames: list[str] = []
    seen: set[str] = set()
    for row in rows:
        for key in row:
            if key not in seen:
                seen.add(key)
                fieldnames.append(key)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def collect_vehicle_metrics(experiments_dir: str | Path) -> list[dict[str, Any]]:
    experiments_dir = Path(experiments_dir)
    runs_dir = experiments_dir / "runs"
    rows: list[dict[str, Any]] = []
    if not runs_dir.exists():
        return rows

    for run_dir in sorted((p for p in runs_dir.iterdir() if p.is_dir()), key=lambda p: _run_order(p.name)):
        metrics_dir = run_dir / "metrics"
        if not metrics_dir.exists():
            continue
        for metric_path in sorted(metrics_dir.glob("*_vehicle_metrics.json")):
            try:
                payload = json.loads(metric_path.read_text(encoding="utf-8"))
            except Exception:
                continue
            if payload.get("level") != "vehicle":
                continue
            payload = dict(payload)
            payload["run_name"] = run_dir.name
            match = re.fullmatch(r"debug_(\d+)", run_dir.name)
            payload["debug_vehicles"] = int(match.group(1)) if match else None
            rows.append(payload)
    return rows


def build_study_summary(config: dict[str, Any]) -> dict[str, Any]:
    experiments_dir = Path(config.get("paths", {}).get("experiments_dir", "experiments"))
    output_dir = experiments_dir / "study_summary"
    tables_dir = output_dir / "tables"
    figures_dir = output_dir / "figures"
    rows = collect_vehicle_metrics(experiments_dir)
    if not rows:
        raise FileNotFoundError(
            f"No se encontraron métricas a nivel vehículo en {experiments_dir / 'runs'}. "
            "Ejecute y evalúe los runs antes de generar el resumen del estudio."
        )

    rows = sorted(rows, key=lambda r: (_run_order(str(r.get("run_name"))), str(r.get("model"))))
    metrics_path = tables_dir / "metrics_across_runs.csv"
    _write_csv(rows, metrics_path)

    best_rows: list[dict[str, Any]] = []
    for run_name in sorted({str(r["run_name"]) for r in rows}, key=_run_order):
        candidates = [r for r in rows if str(r.get("run_name")) == run_name]
        best = max(
            candidates,
            key=lambda r: float(r.get("pr_auc")) if r.get("pr_auc") is not None else -1.0,
        )
        best_rows.append(
            {
                "run_name": run_name,
                "debug_vehicles": best.get("debug_vehicles"),
                "best_model_by_pr_auc": best.get("model"),
                "precision": best.get("precision"),
                "recall": best.get("recall"),
                "f1_score": best.get("f1_score"),
                "pr_auc": best.get("pr_auc"),
                "roc_auc": best.get("roc_auc"),
                "training_time_seconds": best.get("training_time_seconds"),
                "inference_time_seconds": best.get("inference_time_seconds"),
            }
        )
    best_path = tables_dir / "best_model_by_run.csv"
    _write_csv(best_rows, best_path)

    save_cross_run_metric_plot(rows, "pr_auc", figures_dir / "pr_auc_across_runs.png")
    save_cross_run_metric_plot(rows, "f1_score", figures_dir / "f1_across_runs.png")
    save_cross_run_metric_plot(rows, "recall", figures_dir / "recall_across_runs.png")

    completed_runs = sorted({str(r["run_name"]) for r in rows}, key=_run_order)
    summary = {
        "completed_runs": completed_runs,
        "n_vehicle_metric_rows": len(rows),
        "metrics_table": str(metrics_path),
        "best_model_by_run_table": str(best_path),
        "figures": {
            "pr_auc": str(figures_dir / "pr_auc_across_runs.png"),
            "f1_score": str(figures_dir / "f1_across_runs.png"),
            "recall": str(figures_dir / "recall_across_runs.png"),
        },
        "interpretation_note": (
            "Los runs debug documentan validación progresiva y estabilidad del pipeline; "
            "el run full constituye la comparación experimental principal."
        ),
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    summary_path = output_dir / "study_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    return {**summary, "summary_path": str(summary_path)}

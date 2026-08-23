from __future__ import annotations

import csv
import json
import re
from pathlib import Path
from typing import Any

from scania_outliers.visualization import save_learning_curve_plot

_RUN_PATTERN = re.compile(r"learning_curve_n(\d+)_s(\d+)")


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


def _train_vehicle_count(run_dir: Path) -> int | None:
    """Read the actual train vehicle count from a run's own preprocessing table.

    Used for the `full` anchor, whose run-name does not encode a vehicle
    count. `learning_curve_n{N}_s{seed}` runs get N directly from the name,
    which the orchestration script (run_learning_curve.py) guarantees matches
    what was actually sampled.
    """
    summary_path = run_dir / "tables" / "preprocessing" / "window_summary_by_split.csv"
    if not summary_path.exists():
        return None
    with summary_path.open(encoding="utf-8") as f:
        for row in csv.DictReader(f):
            if row.get("split") == "train":
                try:
                    return int(row["n_vehicles"])
                except (KeyError, ValueError, TypeError):
                    return None
    return None


def collect_learning_curve_points(experiments_dir: str | Path, model: str) -> list[dict[str, Any]]:
    """Gather one row per (run, model) with n_vehicles/seed and test-vehicle metrics."""
    experiments_dir = Path(experiments_dir)
    runs_dir = experiments_dir / "runs"
    rows: list[dict[str, Any]] = []
    if not runs_dir.exists():
        return rows

    for run_dir in sorted(p for p in runs_dir.iterdir() if p.is_dir()):
        match = _RUN_PATTERN.fullmatch(run_dir.name)
        if match:
            n_vehicles, seed = int(match.group(1)), int(match.group(2))
            is_anchor = False
        elif run_dir.name == "full":
            n_vehicles = _train_vehicle_count(run_dir)
            seed = None
            is_anchor = True
            if n_vehicles is None:
                continue
        else:
            continue

        metric_path = run_dir / "metrics" / f"{model}_test_vehicle_metrics.json"
        if not metric_path.exists():
            continue
        try:
            payload = json.loads(metric_path.read_text(encoding="utf-8"))
        except Exception:
            continue

        rows.append({
            "run_name": run_dir.name,
            "n_vehicles": n_vehicles,
            "seed": seed if seed is not None else "anchor_full",
            "is_anchor": is_anchor,
            "model": model,
            "precision": payload.get("precision"),
            "recall": payload.get("recall"),
            "f1_score": payload.get("f1_score"),
            "pr_auc": payload.get("pr_auc"),
            "roc_auc": payload.get("roc_auc"),
        })
    return sorted(rows, key=lambda r: (r["n_vehicles"], str(r["seed"])))


def build_learning_curve_summary(config: dict[str, Any]) -> dict[str, Any]:
    experiments_dir = Path(config.get("paths", {}).get("experiments_dir", "experiments"))
    model = str(config.get("learning_curve", {}).get("model", "lstm_autoencoder"))
    output_dir = experiments_dir / "learning_curve_summary"
    tables_dir = output_dir / "tables"
    figures_dir = output_dir / "figures"

    rows = collect_learning_curve_points(experiments_dir, model)
    if not rows:
        raise FileNotFoundError(
            f"No se encontraron runs learning_curve_n*_s* (ni el anchor 'full') con métricas para "
            f"'{model}' en {experiments_dir / 'runs'}. Ejecute scripts/run_learning_curve.py primero."
        )

    n_with_repeats = len({r["n_vehicles"] for r in rows if not r["is_anchor"]})
    table_path = tables_dir / "learning_curve_points.csv"
    _write_csv(rows, table_path)

    metrics_to_plot = ["pr_auc", "f1_score", "recall", "roc_auc"]
    figure_paths: dict[str, str] = {}
    for metric in metrics_to_plot:
        fig_path = figures_dir / f"learning_curve_{metric}.png"
        save_learning_curve_plot(rows, metric, fig_path, model)
        figure_paths[metric] = str(fig_path)

    summary = {
        "model": model,
        "sizes_with_repeats": n_with_repeats,
        "n_points": len(rows),
        "points_table": str(table_path),
        "figures": figure_paths,
        "interpretation_note": (
            "Cada tamaño (salvo el ancla 'full') se repite con varias semillas de muestreo de "
            "vehiculos de train; validation y test se mantienen completos en todos los puntos. "
            "La banda sombreada es +/- una desviacion estandar entre semillas, no un intervalo "
            "de confianza formal."
        ),
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    summary_path = output_dir / "learning_curve_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    return {**summary, "summary_path": str(summary_path)}

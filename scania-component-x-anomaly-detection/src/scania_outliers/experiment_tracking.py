from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path
from typing import Any, Dict, Iterable


def save_json(payload: Dict[str, Any], path: str | Path) -> None:
    """Save an experiment artifact as pretty JSON."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as file:
        json.dump(payload, file, indent=2, ensure_ascii=False, default=str)


def append_metrics_csv(row: Dict[str, Any], path: str | Path) -> None:
    """Append one metric row to a CSV comparison file without Pandas."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    exists = path.exists()
    with path.open("a", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(row.keys()))
        if not exists:
            writer.writeheader()
        writer.writerow(row)


def save_predictions_table(rows: Iterable[Dict[str, Any]], path: str | Path) -> None:
    """Persist window or vehicle-level prediction rows without Pandas."""
    rows = list(rows)
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames: list[str] = []
    for row in rows:
        for key in row.keys():
            if key not in fieldnames:
                fieldnames.append(key)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def file_sha256(path: str | Path, chunk_size: int = 1024 * 1024) -> str:
    """Compute SHA-256 hash for a file to document data/artifact lineage."""
    h = hashlib.sha256()
    path = Path(path)
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(chunk_size), b""):
            h.update(chunk)
    return h.hexdigest()


def save_run_manifest(
    path: str | Path,
    config: dict,
    artifacts: dict | None = None,
    notes: str | None = None,
) -> None:
    """Save a lightweight experiment manifest for MLOps-style traceability."""
    payload = {
        "config": config,
        "artifacts": artifacts or {},
        "notes": notes,
    }
    save_json(payload, path)

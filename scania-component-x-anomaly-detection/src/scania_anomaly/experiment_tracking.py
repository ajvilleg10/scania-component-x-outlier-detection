from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Dict

import pandas as pd


def save_json(payload: Dict[str, Any], path: str | Path) -> None:
    """Save an experiment artifact as pretty JSON."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as file:
        json.dump(payload, file, indent=2, ensure_ascii=False)


def append_metrics_csv(row: Dict[str, Any], path: str | Path) -> None:
    """Append one metric row to a CSV comparison file."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame([row])
    if path.exists():
        df.to_csv(path, mode="a", header=False, index=False)
    else:
        df.to_csv(path, index=False)


def save_predictions_table(df: pd.DataFrame, path: str | Path) -> None:
    """Persist window or vehicle-level predictions."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)


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

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any


def preparation_payload(config: dict[str, Any], run_name: str) -> dict[str, Any]:
    execution = config.get("execution", {})
    preprocessing = config.get("preprocessing", {})
    windowing = config.get("windowing", {})
    dataset = config.get("dataset", {})
    return {
        "run_name": run_name,
        "mode": execution.get("mode", "debug"),
        "max_vehicles": execution.get("max_vehicles_debug") if execution.get("mode", "debug") == "debug" else None,
        "debug_min_positive_vehicles": execution.get("debug_min_positive_vehicles", 2),
        "learning_curve_n_vehicles": execution.get("learning_curve_n_vehicles") if execution.get("mode") == "learning_curve" else None,
        "learning_curve_seed": execution.get("learning_curve_seed") if execution.get("mode") == "learning_curve" else None,
        "dataset_files": dataset.get("files", {}),
        "binary_reference_target": dataset.get("binary_reference_target", {}),
        "preprocessing": {
            "max_missing_ratio": preprocessing.get("max_missing_ratio"),
            "fill_strategy": preprocessing.get("fill_strategy"),
            "drop_constant_columns": preprocessing.get("drop_constant_columns"),
            "exclude_columns": preprocessing.get("exclude_columns", []),
            "scaling_method": preprocessing.get("scaling_method"),
        },
        "windowing": {
            "window_size": windowing.get("window_size"),
            "stride": windowing.get("stride"),
            "label_policy": windowing.get("label_policy"),
            "output_format": windowing.get("output_format"),
        },
    }


def preparation_fingerprint(config: dict[str, Any], run_name: str) -> str:
    payload = preparation_payload(config, run_name)
    raw = json.dumps(payload, sort_keys=True, ensure_ascii=False, separators=(",", ":"))
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()[:20]


def active_preparation_path(config: dict[str, Any]) -> Path:
    processed_dir = Path(config.get("paths", {}).get("processed_dir", "data/processed"))
    return processed_dir / "manifests" / "active_preparation.json"


def write_active_preparation(config: dict[str, Any], run_name: str, extra: dict[str, Any] | None = None) -> Path:
    path = active_preparation_path(config)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = preparation_payload(config, run_name)
    payload["fingerprint"] = preparation_fingerprint(config, run_name)
    if extra:
        payload.update(extra)
    with path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False, default=str)
    return path


def validate_active_preparation(config: dict[str, Any], run_name: str) -> dict[str, Any]:
    path = active_preparation_path(config)
    if not path.exists():
        raise FileNotFoundError(
            "No existe una preparación activa. Ejecute primero --prepare-only con el mismo --run-name."
        )
    with path.open("r", encoding="utf-8") as f:
        payload = json.load(f)

    expected = preparation_fingerprint(config, run_name)
    if payload.get("run_name") != run_name or payload.get("fingerprint") != expected:
        raise RuntimeError(
            "Las ventanas activas no corresponden a esta ejecución. "
            f"Esperado run={run_name}, fingerprint={expected}; "
            f"activo run={payload.get('run_name')}, fingerprint={payload.get('fingerprint')}. "
            "Ejecute --prepare-only para este run antes de entrenar/evaluar."
        )
    return payload

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict

import yaml


def load_config(config_path: str | Path = "config/config.colab.yaml") -> Dict[str, Any]:
    path = Path(config_path)
    if not path.exists():
        raise FileNotFoundError(f"Configuration file not found: {path}")
    with path.open("r", encoding="utf-8") as file:
        return yaml.safe_load(file)


def ensure_directories(config: Dict[str, Any]) -> None:
    """Create only shared runtime roots.

    Run-specific result folders are intentionally created lazily by PipelineContext
    so Drive does not accumulate empty directories that are never used.
    """
    p = config.get("paths", {})
    candidates = [
        p.get("drive_root"),
        p.get("raw_dir"),
        p.get("processed_dir"),
        p.get("experiments_dir"),
    ]
    for value in candidates:
        if not value:
            continue
        try:
            Path(value).mkdir(parents=True, exist_ok=True)
        except Exception:
            pass


def get_dataset_files(config: Dict[str, Any]) -> Dict[str, str]:
    return dict(config.get("dataset", {}).get("files", {}))

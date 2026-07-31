from __future__ import annotations

from pathlib import Path
from typing import Any, Dict

import yaml


def load_config(config_path: str | Path = "config/config.yaml") -> Dict[str, Any]:
    """Load a YAML configuration file."""
    path = Path(config_path)
    if not path.exists():
        raise FileNotFoundError(f"Configuration file not found: {path}")
    with path.open("r", encoding="utf-8") as file:
        return yaml.safe_load(file)


def ensure_directories(config: Dict[str, Any]) -> None:
    """Create Drive/output directories declared in the configuration when available."""
    path_config = config.get("paths", {})
    for key, value in path_config.items():
        if key.endswith("_dir") or key.endswith("_root"):
            try:
                Path(value).mkdir(parents=True, exist_ok=True)
            except Exception:
                # Google Drive paths may not exist before mounting Colab.
                pass


def get_dataset_files(config: Dict[str, Any]) -> Dict[str, str]:
    """Return configured dataset file map."""
    return dict(config.get("dataset", {}).get("files", {}))

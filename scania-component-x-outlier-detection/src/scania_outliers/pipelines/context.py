from __future__ import annotations

import json
import logging
import os
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

from scania_outliers.config import ensure_directories
from scania_outliers.spark_session import create_spark_session


@dataclass
class PipelinePaths:
    """Runtime paths.

    Only raw/processed data are shared between executions. Every model/result
    artifact is written below experiments/runs/<run_id>, which prevents one
    debug/full experiment from overwriting another.
    """

    drive_root: Path
    raw_dir: Path
    processed_dir: Path
    experiments_dir: Path
    run_dir: Path

    @property
    def models_dir(self) -> Path:
        return self.run_dir / "models"

    @property
    def metrics_dir(self) -> Path:
        return self.run_dir / "metrics"

    @property
    def predictions_dir(self) -> Path:
        return self.run_dir / "predictions"

    @property
    def figures_dir(self) -> Path:
        return self.run_dir / "figures"

    @property
    def tables_dir(self) -> Path:
        return self.run_dir / "tables"

    @property
    def comparisons_dir(self) -> Path:
        return self.run_dir / "comparisons"

    @property
    def logs_dir(self) -> Path:
        return self.run_dir / "logs"

    @property
    def manifests_dir(self) -> Path:
        return self.run_dir / "manifests"


class PipelineContext:
    """Central execution context shared by all stages."""

    def __init__(self, config: dict[str, Any], config_path: Path, run_id: str | None = None):
        self.config = config
        self.config_path = config_path
        self.run_id = run_id or time.strftime("run_%Y%m%d_%H%M%S")
        self.logger = self._build_logger()
        ensure_directories(config)
        self.paths = self._build_paths()
        self.paths.run_dir.mkdir(parents=True, exist_ok=True)
        self._save_effective_config()
        self.spark = None

    def _build_logger(self) -> logging.Logger:
        logger = logging.getLogger("scania_outliers")
        logger.setLevel(logging.INFO)
        if not logger.handlers:
            handler = logging.StreamHandler()
            handler.setFormatter(logging.Formatter("[%(asctime)s] %(levelname)s - %(message)s"))
            logger.addHandler(handler)
        return logger

    def _build_paths(self) -> PipelinePaths:
        p = self.config.get("paths", {})
        experiments_dir = Path(p.get("experiments_dir", Path(p.get("drive_root", ".")) / "experiments"))
        return PipelinePaths(
            drive_root=Path(p.get("drive_root", ".")),
            raw_dir=Path(p.get("raw_dir", "data/raw")),
            processed_dir=Path(p.get("processed_dir", "data/processed")),
            experiments_dir=experiments_dir,
            run_dir=experiments_dir / "runs" / self.run_id,
        )

    def _save_effective_config(self) -> None:
        path = self.paths.run_dir / "config_used.yaml"
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", encoding="utf-8") as f:
            yaml.safe_dump(self.config, f, allow_unicode=True, sort_keys=False)

    def get_spark(self):
        if self.spark is None:
            self.logger.info("Creating Spark session")
            self.spark = create_spark_session(self.config)
        return self.spark

    def close(self) -> None:
        if self.spark is not None:
            self.logger.info("Stopping Spark session")
            try:
                self.spark.stop()
            except Exception as exc:
                self.logger.warning("Spark session could not be stopped cleanly: %s", exc)
            finally:
                self.spark = None

    def artifact_path(self, category: str, *parts: str | os.PathLike) -> Path:
        """Return a run-scoped artifact path and create only its parent on demand."""
        path = self.paths.run_dir.joinpath(category, *map(str, parts))
        path.parent.mkdir(parents=True, exist_ok=True)
        return path

    def model_path(self, model_name: str, filename: str) -> Path:
        return self.artifact_path("models", model_name, filename)

    def save_stage_manifest(self, stage: str, payload: dict[str, Any]) -> Path:
        path = self.artifact_path("manifests", f"{stage}.json")
        with path.open("w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2, ensure_ascii=False, default=str)
        return path

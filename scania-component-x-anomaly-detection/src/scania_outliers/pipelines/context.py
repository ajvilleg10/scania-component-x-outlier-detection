from __future__ import annotations

import json
import logging
import os
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from scania_outliers.config import ensure_directories
from scania_outliers.spark_session import create_spark_session


@dataclass
class PipelinePaths:
    drive_root: Path
    raw_dir: Path
    processed_dir: Path
    models_dir: Path
    outputs_dir: Path
    figures_dir: Path
    tables_dir: Path
    metrics_dir: Path
    doc_dir: Path
    run_dir: Path


class PipelineContext:
    """Central object shared by all stages.

    It owns configuration, paths, run id, logging and Spark session creation.
    """

    def __init__(self, config: dict[str, Any], config_path: Path, run_id: str | None = None):
        self.config = config
        self.config_path = config_path
        self.run_id = run_id or time.strftime("run_%Y%m%d_%H%M%S")
        self.logger = self._build_logger()
        ensure_directories(config)
        self.paths = self._build_paths()
        self.paths.run_dir.mkdir(parents=True, exist_ok=True)
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
        outputs_dir = Path(p.get("outputs_dir", "outputs"))
        return PipelinePaths(
            drive_root=Path(p.get("drive_root", ".")),
            raw_dir=Path(p.get("raw_dir", "data/raw")),
            processed_dir=Path(p.get("processed_dir", "data/processed")),
            models_dir=Path(p.get("models_dir", "models")),
            outputs_dir=outputs_dir,
            figures_dir=Path(p.get("figures_dir", outputs_dir / "figures")),
            tables_dir=Path(p.get("tables_dir", outputs_dir / "tables")),
            metrics_dir=Path(p.get("metrics_dir", outputs_dir / "metrics")),
            doc_dir=Path(p.get("doc_dir", "doc")),
            run_dir=outputs_dir / "runs" / self.run_id,
        )

    def get_spark(self):
        if self.spark is None:
            self.logger.info("Creating Spark session")
            self.spark = create_spark_session(self.config)
        return self.spark

    def close(self) -> None:
        if self.spark is not None:
            self.logger.info("Stopping Spark session")
            self.spark.stop()
            self.spark = None

    def artifact_path(self, *parts: str | os.PathLike) -> Path:
        path = self.paths.run_dir.joinpath(*map(str, parts))
        path.parent.mkdir(parents=True, exist_ok=True)
        return path

    def save_stage_manifest(self, stage: str, payload: dict[str, Any]) -> None:
        path = self.artifact_path("manifests", f"{stage}.json")
        with path.open("w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2, ensure_ascii=False, default=str)

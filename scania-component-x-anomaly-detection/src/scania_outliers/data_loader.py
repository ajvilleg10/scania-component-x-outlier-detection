from __future__ import annotations

from pathlib import Path
from typing import Dict, Optional, Any

try:
    from pyspark.sql import DataFrame, SparkSession
except Exception:  # pragma: no cover - lets dry-run/tests import without PySpark
    DataFrame = Any  # type: ignore
    SparkSession = Any  # type: ignore


class ScaniaDataLoader:
    """Loads SCANIA Component X files using PySpark."""

    def __init__(
        self,
        spark: SparkSession,
        raw_dir: str | Path,
        alternative_raw_dir: Optional[str | Path] = None,
        file_map: Optional[Dict[str, str]] = None,
    ):
        self.spark = spark
        self.raw_dir = Path(raw_dir)
        self.alternative_raw_dir = Path(alternative_raw_dir) if alternative_raw_dir else None
        self.file_map = file_map or {}

    @classmethod
    def from_config(cls, spark: SparkSession, config: dict) -> "ScaniaDataLoader":
        return cls(
            spark=spark,
            raw_dir=config["paths"]["raw_dir"],
            alternative_raw_dir=config["paths"].get("raw_dir_alternative"),
            file_map=config.get("dataset", {}).get("files", {}),
        )

    def _resolve(self, filename: str) -> Path:
        candidates = [self.raw_dir / filename]
        if self.alternative_raw_dir is not None:
            candidates.append(self.alternative_raw_dir / filename)

        for candidate in candidates:
            if candidate.exists():
                return candidate

        raise FileNotFoundError(
            "File not found. Checked: " + ", ".join(str(c) for c in candidates)
        )

    def load_csv(self, filename: str, header: bool = True, infer_schema: bool = True) -> DataFrame:
        path = self._resolve(filename)
        return self.spark.read.csv(str(path), header=header, inferSchema=infer_schema)

    def read_csv(self, alias_or_filename: str, header: bool = True, infer_schema: bool = True) -> DataFrame:
        """Load by dataset alias from config or direct filename."""
        filename = self.file_map.get(alias_or_filename, alias_or_filename)
        return self.load_csv(filename, header=header, infer_schema=infer_schema)

    def load_from_config(self, file_map: Dict[str, str], alias: str) -> DataFrame:
        if alias not in file_map:
            raise KeyError(f"Alias '{alias}' not found in dataset file map")
        return self.load_csv(file_map[alias])

    def load_available_files(self, file_map: Dict[str, str] | None = None) -> Dict[str, DataFrame]:
        """Load all configured files that are available."""
        file_map = file_map or self.file_map
        loaded: Dict[str, DataFrame] = {}
        for alias, filename in file_map.items():
            try:
                loaded[alias] = self.load_csv(filename)
            except FileNotFoundError:
                print(f"[WARN] Missing file skipped: {filename}")
        return loaded

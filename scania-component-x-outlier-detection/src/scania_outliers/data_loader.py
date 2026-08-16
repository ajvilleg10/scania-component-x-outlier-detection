from __future__ import annotations

from pathlib import Path
from typing import Dict, Optional, Any

try:
    from pyspark.sql import DataFrame, SparkSession
except Exception:  # pragma: no cover - lets dry-run/tests import without PySpark
    DataFrame = Any  # type: ignore
    SparkSession = Any  # type: ignore


class ScaniaDataLoader:
    """Loads SCANIA Component X files using PySpark.

    The pipeline expects the CSV files to be persisted in Google Drive under
    ``paths.raw_dir``. KaggleHub downloads to a temporary Colab cache, so those
    files must be copied to Drive/data/raw before running the pipeline.
    """

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

    def expected_files(self) -> Dict[str, Path]:
        """Return expected dataset files resolved against the primary raw directory."""
        return {alias: self.raw_dir / filename for alias, filename in self.file_map.items()}

    def missing_files(self, allow_alternative: bool = False) -> Dict[str, str]:
        """Return missing files by alias using the configured primary raw directory."""
        missing: Dict[str, str] = {}
        for alias, filename in self.file_map.items():
            primary = self.raw_dir / filename
            alternative = self.alternative_raw_dir / filename if self.alternative_raw_dir else None
            exists = primary.exists() or (allow_alternative and alternative is not None and alternative.exists())
            if not exists:
                missing[alias] = filename
        return missing

    def validate_required_files(self, allow_alternative: bool = False) -> None:
        missing = self.missing_files(allow_alternative=allow_alternative)
        if missing:
            missing_text = "\n".join(f"- {alias}: {filename}" for alias, filename in missing.items())
            raise FileNotFoundError(
                "Faltan archivos obligatorios del dataset SCANIA Component X en data/raw:\n"
                f"{missing_text}\n\n"
                f"Ruta esperada: {self.raw_dir}\n"
                "KaggleHub descarga inicialmente en /root/.cache/kagglehub/, pero el pipeline no lee desde esa caché.\n"
                "Ejecute primero:\n"
                "python scripts/download_kaggle_to_drive.py --config config/config.colab.yaml\n"
                "python scripts/check_raw_files.py --config config/config.colab.yaml"
            )

    def _resolve(self, filename: str) -> Path:
        candidates = [self.raw_dir / filename]
        if self.alternative_raw_dir is not None:
            candidates.append(self.alternative_raw_dir / filename)

        for candidate in candidates:
            if candidate.exists():
                return candidate

        raise FileNotFoundError(
            "No se encontró el archivo requerido del dataset.\n"
            f"Archivo: {filename}\n"
            "Rutas revisadas:\n"
            + "\n".join(f"- {c}" for c in candidates)
            + "\n\nEl dataset descargado con kagglehub debe copiarse a Google Drive/data/raw antes de ejecutar main.py.\n"
            "Ejecute:\n"
            "python scripts/download_kaggle_to_drive.py --config config/config.colab.yaml\n"
            "python scripts/check_raw_files.py --config config/config.colab.yaml"
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
            except FileNotFoundError as exc:
                print(f"[WARN] Missing file skipped: {filename}. {exc}")
        return loaded

from __future__ import annotations

import csv
import json
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch

from scania_outliers.data_loader import ScaniaDataLoader
from scania_outliers.data_quality import DataQualityAnalyzer, save_quality_report, save_rows_csv
from scania_outliers.datasets import inspect_parquet_windows, make_parquet_loader, score_parquet_windows
from scania_outliers.experiment_tracking import save_json, save_predictions_table
from scania_outliers.labels import (
    attach_vehicle_labels,
    filter_normal_training_vehicles,
    summarize_vehicle_labels,
)
from scania_outliers.model_evaluation import binary_classification_metrics
from scania_outliers.model_factory import ModelFactory
from scania_outliers.outlier_detection import select_threshold, threshold_rows
from scania_outliers.preprocessing import (
    apply_train_fitted_preprocessing,
    get_numeric_columns,
)
from scania_outliers.spark_windowing import SparkWindowBuilder
from scania_outliers.temporal_analysis import time_step_gap_report
from scania_outliers.training.trainer import AutoencoderTrainer, TrainingConfig
from scania_outliers.vehicle_level import (
    aggregate_vehicle_scores,
    classify_vehicle_scores,
)


class BaseStage:
    """Base class for all pipeline stages."""

    def __init__(self, context):
        self.ctx = context
        self.config = context.config
        self.log = context.logger

    @property
    def file_map(self) -> dict[str, str]:
        return self.config.get("dataset", {}).get("files", {})

    @property
    def vehicle_col(self) -> str:
        return self.config.get("dataset", {}).get("vehicle_col", "vehicle_id")

    @property
    def time_col(self) -> str:
        return self.config.get("dataset", {}).get("time_col", "time_step")

    @property
    def label_col(self) -> str:
        return "y_true"

    def loader(self) -> ScaniaDataLoader:
        return ScaniaDataLoader.from_config(self.ctx.get_spark(), self.config)

    def _write_csv_rows(self, rows: list[dict[str, Any]], path: Path) -> None:
        save_rows_csv(rows, path)


class CheckDataStage(BaseStage):
    """Validate that all required CSV files exist in Google Drive/data/raw."""

    def run(self) -> dict[str, Any]:
        self.log.info("Running raw data validation stage")
        raw_dir = Path(self.config["paths"]["raw_dir"])
        alt_raw_value = self.config["paths"].get("raw_dir_alternative")
        alt_raw = Path(alt_raw_value) if alt_raw_value else None
        files = self.file_map
        missing = []
        found = {}

        if not raw_dir.exists():
            raise FileNotFoundError(
                f"No existe el directorio raw esperado: {raw_dir}. "
                "Monte Google Drive y ejecute scripts/create_drive_folders.py."
            )

        for alias, filename in files.items():
            primary = raw_dir / filename
            alternative = alt_raw / filename if alt_raw else None
            if primary.exists():
                found[alias] = str(primary)
            elif alternative and alternative.exists():
                found[alias] = str(alternative)
                self.log.warning("%s encontrado en ruta alternativa: %s", alias, alternative)
            else:
                missing.append((alias, filename))

        if missing:
            missing_text = "\n".join(f"- {alias}: {filename}" for alias, filename in missing)
            raise FileNotFoundError(
                "Faltan archivos obligatorios en Google Drive/data/raw:\n"
                f"{missing_text}\n\n"
                "Ejecute primero:\n"
                "python scripts/download_kaggle_to_drive.py --config config/config.colab.yaml\n"
                "python scripts/check_raw_files.py --config config/config.colab.yaml"
            )

        payload = {
            "raw_dir": str(raw_dir),
            "n_files_found": len(found),
            "files": found,
            "status": "ok",
        }
        manifest_path = self.ctx.artifact_path("manifests", "check_data.json")
        save_json(payload, manifest_path)
        self.log.info("Raw data validation completed. Files found: %s", len(found))
        return payload


class EDAStage(BaseStage):
    """Exploratory analysis and quality control stage implemented with Spark only."""

    def run(self) -> dict[str, Any]:
        self.log.info("Running EDA and data quality stage")
        loader = self.loader()
        loaded = loader.load_available_files(self.file_map)
        summary: dict[str, Any] = {"loaded_files": list(loaded.keys()), "files": {}}

        for alias, df in loaded.items():
            analyzer = DataQualityAnalyzer(df)
            n_rows, n_cols = analyzer.shape()
            file_summary = {"n_rows": n_rows, "n_cols": n_cols}
            self.log.info("%s: %s rows, %s columns", alias, n_rows, n_cols)

            schema_path = self.ctx.artifact_path("tables", f"schema_{alias}.csv")
            save_rows_csv(analyzer.schema_rows(), schema_path, fieldnames=["column", "dtype"])
            file_summary["schema_path"] = str(schema_path)

            try:
                missing = analyzer.missing_report()
                missing_path = self.ctx.artifact_path("tables", f"missing_{alias}.csv")
                save_quality_report(missing, str(missing_path))
                file_summary["missing_report_path"] = str(missing_path)
            except Exception as exc:  # pragma: no cover - depends on Spark/data availability
                file_summary["missing_report_error"] = str(exc)

            if self.vehicle_col in df.columns and self.time_col in df.columns:
                try:
                    temporal = time_step_gap_report(df, self.vehicle_col, self.time_col)
                    temporal_path = self.ctx.artifact_path("tables", f"temporal_gaps_{alias}.csv")
                    save_rows_csv(temporal, temporal_path)
                    file_summary["temporal_gap_report_path"] = str(temporal_path)
                except Exception as exc:  # pragma: no cover
                    file_summary["temporal_gap_error"] = str(exc)

            if alias.endswith("labels") or alias == "train_tte":
                try:
                    label_summary = summarize_vehicle_labels(df, vehicle_col=self.vehicle_col)
                    file_summary["label_summary"] = label_summary
                except Exception as exc:
                    file_summary["label_summary_error"] = str(exc)

            summary["files"][alias] = file_summary

        manifest_path = self.ctx.artifact_path("manifests", "eda.json")
        save_json(summary, manifest_path)
        return summary


class PreprocessingWindowingStage(BaseStage):
    """Leakage-safe preprocessing and Spark-native window construction."""

    def _max_vehicles(self) -> int | None:
        execution = self.config.get("execution", {})
        if execution.get("mode", "debug") == "debug":
            return int(execution.get("max_vehicles_debug", 100))
        return None

    def _windows_dir(self, split: str) -> Path:
        return self.ctx.paths.processed_dir / "windows" / split

    def run(self) -> dict[str, Any]:
        self.log.info("Running preprocessing and Spark-native windowing stage")
        loader = self.loader()

        train_op = loader.read_csv("train_operational")
        validation_op = loader.read_csv("validation_operational")
        test_op = loader.read_csv("test_operational")
        train_tte = loader.read_csv("train_tte")
        validation_labels = loader.read_csv("validation_labels")
        test_labels = loader.read_csv("test_labels")

        train_normal = filter_normal_training_vehicles(train_op, train_tte, vehicle_col=self.vehicle_col)
        validation_labeled = attach_vehicle_labels(validation_op, validation_labels, vehicle_col=self.vehicle_col)
        test_labeled = attach_vehicle_labels(test_op, test_labels, vehicle_col=self.vehicle_col)

        pre_cfg = self.config.get("preprocessing", {})
        exclude = pre_cfg.get("exclude_columns", [])
        feature_cols = get_numeric_columns(train_normal, exclude=exclude)
        self.log.info("Candidate numeric feature columns: %s", len(feature_cols))

        analyzer = DataQualityAnalyzer(train_normal)
        missing_report = analyzer.missing_report()
        constant_cols = analyzer.constant_columns(feature_cols) if pre_cfg.get("drop_constant_columns", True) else []

        train_proc, val_proc, test_proc, scaled_cols, metadata = apply_train_fitted_preprocessing(
            train_df=train_normal,
            validation_df=validation_labeled,
            test_df=test_labeled,
            feature_cols=feature_cols,
            missing_report=missing_report,
            constant_cols=constant_cols,
            max_missing_ratio=float(pre_cfg.get("max_missing_ratio", 0.8)),
            vehicle_col=self.vehicle_col,
            time_col=self.time_col,
            label_col=self.label_col,
        )

        w_cfg = self.config.get("windowing", {})
        builder = SparkWindowBuilder(
            vehicle_col=self.vehicle_col,
            time_col=self.time_col,
            window_size=int(w_cfg.get("window_size", 30)),
            stride=int(w_cfg.get("stride", 5)),
            label_policy=w_cfg.get("label_policy", "vehicle_label"),
        )
        max_vehicles = self._max_vehicles()
        label_col = self.label_col
        partitions = int(w_cfg.get("parquet_partitions", self.config.get("spark", {}).get("shuffle_partitions", 8)))

        split_dfs = {
            "train": train_proc,
            "validation": val_proc,
            "test": test_proc,
        }
        window_paths: dict[str, str] = {}
        split_metadata: dict[str, dict[str, Any]] = {}
        for split, df in split_dfs.items():
            self.log.info("Building %s windows with Spark", split)
            windows_df = builder.build(df, scaled_cols, max_vehicles=max_vehicles, label_col=label_col)
            output_dir = self._windows_dir(split)
            meta = builder.write_parquet(windows_df, output_dir, partitions=partitions)
            window_paths[f"{split}_windows"] = str(output_dir)
            split_metadata[split] = meta.to_dict()
            self.log.info("%s windows saved to %s (%s windows)", split, output_dir, meta.n_windows)

        metadata_payload = metadata.to_dict()
        metadata_payload.update(
            {
                "window_format": "parquet",
                "use_pandas": False,
                "splits": split_metadata,
                "window_size": int(w_cfg.get("window_size", 30)),
                "stride": int(w_cfg.get("stride", 5)),
                "label_policy": w_cfg.get("label_policy", "vehicle_label"),
                "execution_mode": self.config.get("execution", {}).get("mode", "debug"),
                "max_vehicles": max_vehicles,
            }
        )
        metadata_path = self.ctx.artifact_path("manifests", pre_cfg.get("metadata_file", "preprocessing_metadata.json"))
        save_json(metadata_payload, metadata_path)

        return {"window_paths": window_paths, "metadata_path": str(metadata_path), **metadata_payload}


class TrainingStage(BaseStage):
    """Model training stage for reconstruction-based outlier detection."""

    def _window_path(self, split: str) -> Path:
        return self.ctx.paths.processed_dir / "windows" / split

    def _make_loader(self, split: str, batch_size: int):
        return make_parquet_loader(self._window_path(split), batch_size=batch_size)

    def run(self, model: str = "all") -> dict[str, Any]:
        self.log.info("Running training stage")
        modeling = self.config.get("modeling", {})
        requested_models = ModelFactory.resolve_requested_models(model, self.config)
        batch_size = int(modeling.get("batch_size", 128))

        train_info = inspect_parquet_windows(self._window_path("train"))
        n_features = int(train_info["n_features"])
        window_size = int(train_info["window_size"])
        if n_features <= 0 or window_size <= 0:
            raise ValueError("Invalid train window metadata. Run preprocessing/windowing first.")

        # The SCANIA Component X dataset already provides official train, validation and test partitions.
        # Therefore, the pipeline does not create k-fold splits or any additional training partitions.
        train_loader = self._make_loader("train", batch_size=batch_size)
        validation_loader = self._make_loader("validation", batch_size=batch_size)

        out: dict[str, Any] = {"models": {}, "window_metadata": train_info}
        for model_name in requested_models:
            self.log.info("Training model: %s", model_name)
            model_obj = ModelFactory.create(model_name, n_features=n_features, window_size=window_size, config=self.config)
            trainer = AutoencoderTrainer(
                model_obj,
                TrainingConfig(
                    epochs=int(modeling.get("epochs", 20)),
                    learning_rate=float(modeling.get("learning_rate", 1e-3)),
                    batch_size=batch_size,
                    patience=int(modeling.get("early_stopping_patience", 5)),
                    device=str(modeling.get("device", "auto")),
                ),
            )
            history = trainer.fit(train_loader, validation_loader)
            model_dir = self.ctx.paths.models_dir / model_name
            model_path = model_dir / "model.pt"
            trainer.save(model_path)

            validation_rows = score_parquet_windows(model_obj, self._window_path("validation"), batch_size=batch_size, device=str(modeling.get("device", "auto")))
            validation_vehicle_scores = aggregate_vehicle_scores(validation_rows)
            score_col = self.config.get("outlier_detection", {}).get("vehicle_aggregation", {}).get("primary_score", "max_score")
            threshold = select_threshold(
                scores=np.asarray([r[score_col] for r in validation_vehicle_scores], dtype=float),
                y_true=np.asarray([r["y_true"] for r in validation_vehicle_scores], dtype=int),
                strategy=self.config.get("outlier_detection", {}).get("threshold_strategy", "best_f1_on_validation"),
                percentile=float(self.config.get("outlier_detection", {}).get("threshold_percentile", 95)),
            )

            threshold_payload = {
                "model": model_name,
                "threshold": float(threshold),
                "selection_split": "validation",
                "selection_level": "vehicle",
                "score_col": score_col,
                "history": history,
            }
            threshold_path = model_dir / "threshold.json"
            save_json(threshold_payload, threshold_path)
            history_path = model_dir / "training_history.json"
            save_json({"model": model_name, "history": history}, history_path)

            out["models"][model_name] = {
                "model_path": str(model_path),
                "threshold_path": str(threshold_path),
                "training_history_path": str(history_path),
                "threshold": float(threshold),
            }

        self.ctx.save_stage_manifest("train", out)
        return out


class EvaluationStage(BaseStage):
    """Evaluation stage using validation-selected thresholds and final test split."""

    def _window_path(self, split: str) -> Path:
        return self.ctx.paths.processed_dir / "windows" / split

    def _load_threshold(self, model_name: str) -> dict[str, Any]:
        path = self.ctx.paths.models_dir / model_name / "threshold.json"
        if not path.exists():
            raise FileNotFoundError(f"Threshold file not found for {model_name}: {path}")
        with path.open("r", encoding="utf-8") as f:
            return json.load(f)

    def _load_model(self, model_name: str, n_features: int, window_size: int):
        model = ModelFactory.create(model_name, n_features=n_features, window_size=window_size, config=self.config)
        model_path = self.ctx.paths.models_dir / model_name / "model.pt"
        state = torch.load(model_path, map_location="cpu")
        model.load_state_dict(state)
        return model

    def run(self, model: str = "all", split: str | None = None) -> dict[str, Any]:
        self.log.info("Running evaluation stage")
        requested_models = ModelFactory.resolve_requested_models(model, self.config)
        eval_cfg = self.config.get("evaluation", {})
        split = split or eval_cfg.get("primary_split", "test")
        batch_size = int(self.config.get("modeling", {}).get("batch_size", 128))
        device = str(self.config.get("modeling", {}).get("device", "auto"))

        window_info = inspect_parquet_windows(self._window_path(split))
        n_features = int(window_info["n_features"])
        window_size = int(window_info["window_size"])

        all_metrics: list[dict[str, Any]] = []
        output: dict[str, Any] = {"split": split, "models": {}}
        for model_name in requested_models:
            self.log.info("Evaluating model: %s", model_name)
            threshold_payload = self._load_threshold(model_name)
            threshold = float(threshold_payload["threshold"])
            vehicle_score_col = threshold_payload.get("score_col", "max_score")
            model_obj = self._load_model(model_name, n_features=n_features, window_size=window_size)

            start = time.time()
            window_rows = score_parquet_windows(model_obj, self._window_path(split), batch_size=batch_size, device=device)
            inference_time = round(time.time() - start, 3)
            window_rows = threshold_rows(window_rows, threshold, score_col="outlier_score", pred_col="is_outlier")
            vehicle_rows = aggregate_vehicle_scores(window_rows)
            vehicle_rows = classify_vehicle_scores(vehicle_rows, threshold, score_col=vehicle_score_col)

            vehicle_metrics = binary_classification_metrics(
                [r["y_true"] for r in vehicle_rows],
                [r["is_outlier"] for r in vehicle_rows],
                scores=[r[vehicle_score_col] for r in vehicle_rows],
            )
            window_metrics = binary_classification_metrics(
                [r["y_true"] for r in window_rows],
                [r["is_outlier"] for r in window_rows],
                scores=[r["outlier_score"] for r in window_rows],
            )
            vehicle_metrics.update({"model": model_name, "split": split, "level": "vehicle", "threshold": threshold, "inference_time_seconds": inference_time})
            window_metrics.update({"model": model_name, "split": split, "level": "window", "threshold": threshold, "inference_time_seconds": inference_time})
            all_metrics.extend([vehicle_metrics, window_metrics])

            pred_dir = self.ctx.paths.outputs_dir / "predictions" / model_name
            save_predictions_table(window_rows, pred_dir / f"{split}_window_predictions.csv")
            save_predictions_table(vehicle_rows, pred_dir / f"{split}_vehicle_predictions.csv")
            save_json(vehicle_metrics, self.ctx.paths.metrics_dir / f"{model_name}_{split}_vehicle_metrics.json")
            save_json(window_metrics, self.ctx.paths.metrics_dir / f"{model_name}_{split}_window_metrics.json")
            output["models"][model_name] = {"vehicle_metrics": vehicle_metrics, "window_metrics": window_metrics}

        comparison_path = self.ctx.paths.metrics_dir / f"comparison_{split}.csv"
        save_rows_csv(all_metrics, comparison_path)
        output["comparison_path"] = str(comparison_path)
        self.ctx.save_stage_manifest("evaluate", output)
        return output


class ComparisonStage(BaseStage):
    """Creates a consolidated metrics table from previous evaluation artifacts."""

    def run(self) -> dict[str, Any]:
        metrics_dir = self.ctx.paths.metrics_dir
        files = sorted(metrics_dir.glob("*_metrics.json"))
        rows = []
        for path in files:
            with path.open("r", encoding="utf-8") as f:
                rows.append(json.load(f))
        if not rows:
            return {"message": "No metric JSON files found", "metrics_dir": str(metrics_dir)}
        out_path = metrics_dir / "comparison_all_models.csv"
        save_rows_csv(rows, out_path)
        return {"comparison_path": str(out_path), "n_rows": len(rows)}

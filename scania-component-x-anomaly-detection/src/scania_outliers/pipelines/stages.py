from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

from scania_outliers.data_loader import ScaniaDataLoader
from scania_outliers.data_quality import DataQualityAnalyzer, save_quality_report
from scania_outliers.datasets import WindowDataset, to_tensor_dataset
from scania_outliers.experiment_tracking import save_json, save_predictions_table
from scania_outliers.labels import (
    attach_vehicle_labels,
    filter_normal_training_vehicles,
    summarize_vehicle_labels,
)
from scania_outliers.model_evaluation import binary_classification_metrics
from scania_outliers.model_factory import ModelFactory
from scania_outliers.outlier_detection import classify_outliers, reconstruction_errors, select_threshold
from scania_outliers.preprocessing import (
    apply_train_fitted_preprocessing,
    get_numeric_columns,
)
from scania_outliers.temporal_analysis import time_step_gap_report
from scania_outliers.training.trainer import AutoencoderTrainer, TrainingConfig
from scania_outliers.vehicle_level import (
    aggregate_vehicle_scores,
    classify_vehicle_scores,
    make_window_predictions,
)
from scania_outliers.windowing import TimeWindowBuilder, WindowData


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


class EDAStage(BaseStage):
    """Exploratory analysis and quality control stage."""

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
            analyzer.schema_as_pandas().to_csv(schema_path, index=False)
            file_summary["schema_path"] = str(schema_path)

            # Missing reports over very large files can be expensive, but they are part of the methodology.
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
                    temporal.to_csv(temporal_path, index=False)
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
    """Leakage-safe preprocessing and window construction."""

    def _max_vehicles(self) -> int | None:
        execution = self.config.get("execution", {})
        if execution.get("mode", "debug") == "debug":
            return int(execution.get("max_vehicles_debug", 100))
        return None

    def run(self) -> dict[str, Any]:
        self.log.info("Running preprocessing and windowing stage")
        loader = self.loader()
        f = self.file_map

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
        builder = TimeWindowBuilder(
            vehicle_col=self.vehicle_col,
            time_col=self.time_col,
            window_size=int(w_cfg.get("window_size", 30)),
            stride=int(w_cfg.get("stride", 5)),
            label_policy=w_cfg.get("label_policy", "vehicle_label"),
        )
        max_vehicles = self._max_vehicles()
        label_col = self.label_col

        train_windows = builder.build_from_spark(train_proc, scaled_cols, max_vehicles=max_vehicles, label_col=label_col)
        validation_windows = builder.build_from_spark(val_proc, scaled_cols, max_vehicles=max_vehicles, label_col=label_col)
        test_windows = builder.build_from_spark(test_proc, scaled_cols, max_vehicles=max_vehicles, label_col=label_col)

        processed_dir = self.ctx.paths.processed_dir
        processed_dir.mkdir(parents=True, exist_ok=True)
        paths = {
            "train_windows": processed_dir / "train_windows.npz",
            "validation_windows": processed_dir / "validation_windows.npz",
            "test_windows": processed_dir / "test_windows.npz",
        }
        builder.save_npz(paths["train_windows"], train_windows)
        builder.save_npz(paths["validation_windows"], validation_windows)
        builder.save_npz(paths["test_windows"], test_windows)

        metadata_payload = metadata.to_dict()
        metadata_payload.update(
            {
                "n_train_windows": int(len(train_windows.X)),
                "n_validation_windows": int(len(validation_windows.X)),
                "n_test_windows": int(len(test_windows.X)),
                "window_size": int(w_cfg.get("window_size", 30)),
                "stride": int(w_cfg.get("stride", 5)),
                "label_policy": w_cfg.get("label_policy", "vehicle_label"),
                "execution_mode": self.config.get("execution", {}).get("mode", "debug"),
                "max_vehicles": max_vehicles,
            }
        )
        metadata_path = self.ctx.artifact_path("manifests", pre_cfg.get("metadata_file", "preprocessing_metadata.json"))
        save_json(metadata_payload, metadata_path)

        return {"window_paths": {k: str(v) for k, v in paths.items()}, "metadata_path": str(metadata_path), **metadata_payload}


class TrainingStage(BaseStage):
    """Model training stage for reconstruction-based outlier detection."""

    def _window_path(self, split: str) -> Path:
        return self.ctx.paths.processed_dir / f"{split}_windows.npz"

    def _make_loader(self, window_data: WindowData, batch_size: int, shuffle: bool = False) -> DataLoader:
        return DataLoader(to_tensor_dataset(window_data), batch_size=batch_size, shuffle=shuffle)

    def run(self, model: str = "all") -> dict[str, Any]:
        self.log.info("Running training stage")
        modeling = self.config.get("modeling", {})
        requested_models = ModelFactory.resolve_requested_models(model, self.config)
        batch_size = int(modeling.get("batch_size", 128))
        train_windows = TimeWindowBuilder.load_npz(self._window_path("train"))
        validation_windows = TimeWindowBuilder.load_npz(self._window_path("validation"))
        n_features = int(train_windows.X.shape[-1])
        window_size = int(train_windows.X.shape[1])

        # The SCANIA Component X dataset already provides official train, validation and test partitions.
        # Therefore, the pipeline does not create k-fold splits or any additional training partitions.
        # Train is used to fit model weights; official validation controls early stopping and threshold selection;
        # test remains reserved for final evaluation.
        train_loader = self._make_loader(train_windows, batch_size=batch_size, shuffle=True)
        validation_loader = self._make_loader(validation_windows, batch_size=batch_size, shuffle=False)
        val_loss_loader = validation_loader

        out: dict[str, Any] = {"models": {}}
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
            history = trainer.fit(train_loader, val_loss_loader)
            model_dir = self.ctx.paths.models_dir / model_name
            model_path = model_dir / "model.pt"
            trainer.save(model_path)

            validation_scores = reconstruction_errors(model_obj, validation_loader, device=str(modeling.get("device", "auto")))
            validation_window_predictions = make_window_predictions(
                vehicle_ids=validation_windows.vehicle_ids,
                y_true=validation_windows.y,
                scores=validation_scores,
                start_time=validation_windows.start_time,
                end_time=validation_windows.end_time,
            )
            validation_vehicle_scores = aggregate_vehicle_scores(validation_window_predictions)
            score_col = self.config.get("outlier_detection", {}).get("vehicle_aggregation", {}).get("primary_score", "max_score")
            threshold = select_threshold(
                scores=validation_vehicle_scores[score_col].to_numpy(),
                y_true=validation_vehicle_scores["y_true"].to_numpy(),
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
        return self.ctx.paths.processed_dir / f"{split}_windows.npz"

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

        window_data = TimeWindowBuilder.load_npz(self._window_path(split))
        loader = DataLoader(to_tensor_dataset(window_data), batch_size=batch_size, shuffle=False)
        n_features = int(window_data.X.shape[-1])
        window_size = int(window_data.X.shape[1])

        all_metrics: list[dict[str, Any]] = []
        output: dict[str, Any] = {"split": split, "models": {}}
        for model_name in requested_models:
            self.log.info("Evaluating model: %s", model_name)
            threshold_payload = self._load_threshold(model_name)
            threshold = float(threshold_payload["threshold"])
            vehicle_score_col = threshold_payload.get("score_col", "max_score")
            model_obj = self._load_model(model_name, n_features=n_features, window_size=window_size)

            start = time.time()
            scores = reconstruction_errors(model_obj, loader, device=device)
            inference_time = round(time.time() - start, 3)
            window_pred = classify_outliers(scores, threshold)
            window_df = make_window_predictions(
                vehicle_ids=window_data.vehicle_ids,
                scores=scores,
                y_true=window_data.y,
                predictions=window_pred,
                start_time=window_data.start_time,
                end_time=window_data.end_time,
            )
            vehicle_df = aggregate_vehicle_scores(window_df)
            vehicle_df = classify_vehicle_scores(vehicle_df, threshold, score_col=vehicle_score_col)

            vehicle_metrics = binary_classification_metrics(
                vehicle_df["y_true"].to_numpy(),
                vehicle_df["is_outlier"].to_numpy(),
                scores=vehicle_df[vehicle_score_col].to_numpy(),
            )
            window_metrics = binary_classification_metrics(
                window_df["y_true"].to_numpy(),
                window_df["is_outlier"].to_numpy(),
                scores=window_df["outlier_score"].to_numpy(),
            )
            vehicle_metrics.update({"model": model_name, "split": split, "level": "vehicle", "threshold": threshold, "inference_time_seconds": inference_time})
            window_metrics.update({"model": model_name, "split": split, "level": "window", "threshold": threshold, "inference_time_seconds": inference_time})
            all_metrics.extend([vehicle_metrics, window_metrics])

            pred_dir = self.ctx.paths.outputs_dir / "predictions" / model_name
            save_predictions_table(window_df, pred_dir / f"{split}_window_predictions.csv")
            save_predictions_table(vehicle_df, pred_dir / f"{split}_vehicle_predictions.csv")
            save_json(vehicle_metrics, self.ctx.paths.metrics_dir / f"{model_name}_{split}_vehicle_metrics.json")
            save_json(window_metrics, self.ctx.paths.metrics_dir / f"{model_name}_{split}_window_metrics.json")
            output["models"][model_name] = {"vehicle_metrics": vehicle_metrics, "window_metrics": window_metrics}

        comparison = pd.DataFrame(all_metrics)
        comparison_path = self.ctx.paths.metrics_dir / f"comparison_{split}.csv"
        comparison_path.parent.mkdir(parents=True, exist_ok=True)
        comparison.to_csv(comparison_path, index=False)
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
        df = pd.DataFrame(rows)
        out_path = metrics_dir / "comparison_all_models.csv"
        df.to_csv(out_path, index=False)
        return {"comparison_path": str(out_path), "n_rows": len(df)}

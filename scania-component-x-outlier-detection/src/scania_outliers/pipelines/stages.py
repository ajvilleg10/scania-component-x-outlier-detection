from __future__ import annotations

import csv
import json
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
try:
    from pyspark.sql import functions as F
    from pyspark.storagelevel import StorageLevel
except Exception:  # pragma: no cover
    F = None  # type: ignore
    StorageLevel = None  # type: ignore

from scania_outliers.data_loader import ScaniaDataLoader
from scania_outliers.data_quality import DataQualityAnalyzer, save_quality_report, save_rows_csv
from scania_outliers.datasets import inspect_parquet_windows, make_parquet_loader, score_parquet_windows
from scania_outliers.experiment_tracking import save_json, save_predictions_table
from scania_outliers.labels import (
    attach_vehicle_labels,
    filter_normal_training_vehicles,
    prepare_vehicle_level_labels,
    summarize_vehicle_labels,
)
from scania_outliers.model_evaluation import binary_classification_metrics
from scania_outliers.model_factory import ModelFactory
from scania_outliers.outlier_detection import select_threshold, threshold_rows
from scania_outliers.preprocessing import (
    apply_train_fitted_preprocessing,
    constant_columns_fast,
    get_numeric_columns,
)
from scania_outliers.run_state import validate_active_preparation, write_active_preparation
from scania_outliers.spark_windowing import SparkWindowBuilder
from scania_outliers.temporal_analysis import time_step_gap_report, trajectory_length_report
from scania_outliers.training.trainer import AutoencoderTrainer, TrainingConfig
from scania_outliers.vehicle_level import aggregate_vehicle_scores, classify_vehicle_scores
from scania_outliers.visualization import (
    save_confusion_matrix_plot,
    save_correlation_matrix,
    save_feature_boxplots,
    save_feature_distributions,
    save_histogram,
    save_label_distribution_plot,
    save_missing_values_plot,
    save_model_comparison_plot,
    save_original_class_distribution_plot,
    save_precision_recall_curve,
    save_preprocessing_feature_summary,
    save_roc_curve,
    save_runtime_comparison_plot,
    save_score_boxplot,
    save_score_distribution,
    save_training_history_plot,
    save_windows_by_split_plot,
)


class BaseStage:
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

    def _run_table(self, section: str, filename: str) -> Path:
        return self.ctx.artifact_path("tables", section, filename)

    def _run_figure(self, section: str, filename: str) -> Path:
        return self.ctx.artifact_path("figures", section, filename)


class CheckDataStage(BaseStage):
    """Validate all configured raw files in Drive/data/raw."""

    def run(self) -> dict[str, Any]:
        self.log.info("Running raw data validation stage")
        raw_dir = Path(self.config["paths"]["raw_dir"])
        alt_raw_value = self.config["paths"].get("raw_dir_alternative")
        alt_raw = Path(alt_raw_value) if alt_raw_value else None
        missing: list[tuple[str, str]] = []
        found: dict[str, str] = {}

        if not raw_dir.exists():
            raise FileNotFoundError(f"No existe el directorio raw esperado: {raw_dir}")

        for alias, filename in self.file_map.items():
            primary = raw_dir / filename
            alternative = alt_raw / filename if alt_raw else None
            if primary.exists():
                found[alias] = str(primary)
            elif alternative and alternative.exists():
                found[alias] = str(alternative)
            else:
                missing.append((alias, filename))

        if missing:
            text = "\n".join(f"- {a}: {f}" for a, f in missing)
            raise FileNotFoundError(f"Faltan archivos obligatorios:\n{text}")

        payload = {"raw_dir": str(raw_dir), "n_files_found": len(found), "files": found, "status": "ok"}
        save_json(payload, self.ctx.artifact_path("manifests", "check_data.json"))
        self.log.info("Raw data validation completed. Files found: %s", len(found))
        return payload


class EDAStage(BaseStage):
    """Spark-based EDA with bounded samples only for visualization.

    No ``toPandas`` is used. Expensive statistics are computed with Spark, while
    Matplotlib receives only small, explicitly bounded arrays intended for plots.
    """

    def _fast_debug(self) -> bool:
        return self.config.get("execution", {}).get("mode", "debug") == "debug" and bool(
            self.config.get("eda", {}).get("fast_debug", True)
        )

    @staticmethod
    def _is_operational(alias: str) -> bool:
        return alias in {"train_operational", "validation_operational", "test_operational"}

    def _numeric_plot_columns(self, df) -> list[str]:
        cfg = self.config.get("eda", {})
        exclude = {self.vehicle_col, self.time_col, "y_true", "class_label", "in_study_repair"}
        numeric = get_numeric_columns(df, exclude=list(exclude))
        return numeric[: int(cfg.get("visual_max_features", 10))]

    def _collect_visual_sample(self, df, columns: list[str]) -> np.ndarray:
        if not columns:
            return np.empty((0, 0), dtype=float)
        cfg = self.config.get("eda", {})
        fraction = float(cfg.get("visual_sample_fraction", 0.02))
        max_rows = int(cfg.get("visual_max_rows", 5000))
        selected = [F.col(c).cast("double").alias(c) for c in columns]
        sampled = df.select(*selected).sample(withReplacement=False, fraction=fraction, seed=42).limit(max_rows)
        rows = sampled.collect()
        if not rows:
            rows = df.select(*selected).limit(min(max_rows, 1000)).collect()
        matrix = []
        for row in rows:
            matrix.append([float(row[c]) if row[c] is not None else np.nan for c in columns])
        return np.asarray(matrix, dtype=float) if matrix else np.empty((0, len(columns)), dtype=float)

    def _trajectory_counts_sample(self, df) -> list[float]:
        max_vehicles = int(self.config.get("eda", {}).get("trajectory_plot_max_vehicles", 5000))
        rows = df.groupBy(self.vehicle_col).count().limit(max_vehicles).collect()
        return [float(r["count"]) for r in rows]

    def run(self) -> dict[str, Any]:
        self.log.info("Running EDA and data quality stage")
        loader = self.loader()
        loaded = loader.load_available_files(self.file_map)
        summary: dict[str, Any] = {"loaded_files": list(loaded.keys()), "files": {}, "fast_debug": self._fast_debug()}
        label_counts: dict[str, dict[int, int]] = {}
        original_class_counts: dict[str, dict[int, int]] = {}

        for alias, df in loaded.items():
            analyzer = DataQualityAnalyzer(df)
            skip_heavy = self._fast_debug() and self._is_operational(alias)
            file_summary: dict[str, Any] = {"n_cols": len(df.columns)}

            if skip_heavy and bool(self.config.get("eda", {}).get("skip_exact_counts_in_debug", True)):
                file_summary["n_rows"] = "skipped_fast_debug"
            else:
                n_rows, n_cols = analyzer.shape()
                file_summary.update({"n_rows": n_rows, "n_cols": n_cols})

            schema_path = self._run_table("eda", f"schema_{alias}.csv")
            save_rows_csv(analyzer.schema_rows(), schema_path, fieldnames=["column", "dtype"])
            file_summary["schema_path"] = str(schema_path)

            missing: list[dict] | None = None
            if skip_heavy:
                file_summary["missing_report_skipped"] = "fast_debug"
                file_summary["temporal_gap_report_skipped"] = "fast_debug"
            else:
                try:
                    missing = analyzer.missing_report()
                    missing_path = self._run_table("eda", f"missing_{alias}.csv")
                    save_quality_report(missing, str(missing_path))
                    file_summary["missing_report_path"] = str(missing_path)
                    save_missing_values_plot(
                        missing,
                        self._run_figure("eda", f"missing_values_{alias}.png"),
                        top_n=int(self.config.get("eda", {}).get("missing_plot_top_n", 25)),
                    )
                except Exception as exc:
                    file_summary["missing_report_error"] = str(exc)

                if self.vehicle_col in df.columns and self.time_col in df.columns:
                    try:
                        temporal = time_step_gap_report(df, self.vehicle_col, self.time_col)
                        temporal_path = self._run_table("eda", f"temporal_gaps_{alias}.csv")
                        save_rows_csv(temporal, temporal_path)
                        file_summary["temporal_gap_report_path"] = str(temporal_path)
                        lengths = trajectory_length_report(df, self.vehicle_col)
                        length_path = self._run_table("eda", f"trajectory_length_summary_{alias}.csv")
                        save_rows_csv(lengths, length_path)
                    except Exception as exc:
                        file_summary["temporal_report_error"] = str(exc)

            if alias.endswith("labels") or alias == "train_tte":
                try:
                    if "class_label" in df.columns:
                        raw_rows = df.groupBy(F.col("class_label").cast("int").alias("class_label")).count().collect()
                        original_class_counts[alias] = {
                            int(r["class_label"]): int(r["count"])
                            for r in raw_rows
                            if r["class_label"] is not None
                        }
                        file_summary["original_class_distribution"] = original_class_counts[alias]

                    label_summary = summarize_vehicle_labels(df, vehicle_col=self.vehicle_col)
                    file_summary["binary_label_summary"] = label_summary
                    prepared = prepare_vehicle_level_labels(df, vehicle_col=self.vehicle_col)
                    rows = prepared.groupBy("y_true").count().collect()
                    label_counts[alias] = {int(r["y_true"]): int(r["count"]) for r in rows if r["y_true"] is not None}
                except Exception as exc:
                    file_summary["label_summary_error"] = str(exc)

            # The detailed distribution/box/correlation figures are generated only
            # for training operational data to avoid redundant wide scans.
            if alias == "train_operational" and bool(self.config.get("eda", {}).get("generate_feature_figures", True)):
                try:
                    feature_cols = self._numeric_plot_columns(df)
                    values = self._collect_visual_sample(df, feature_cols)
                    save_feature_distributions(values, feature_cols[:6], self._run_figure("eda", "train_feature_distributions.png"))
                    save_feature_boxplots(values[:, : min(8, values.shape[1])], feature_cols[:8], self._run_figure("eda", "train_feature_boxplots.png"))
                    corr_n = min(int(self.config.get("eda", {}).get("correlation_max_features", 12)), values.shape[1])
                    save_correlation_matrix(values[:, :corr_n], feature_cols[:corr_n], self._run_figure("eda", "train_feature_correlation_matrix.png"))
                    save_rows_csv(
                        [{"feature": c, "visualization_sample": True} for c in feature_cols],
                        self._run_table("eda", "visualized_train_features.csv"),
                    )
                    trajectory_counts = self._trajectory_counts_sample(df)
                    save_histogram(
                        trajectory_counts,
                        self._run_figure("eda", "train_records_per_vehicle_distribution.png"),
                        title="Distribución del número de lecturas por vehículo",
                        xlabel="Lecturas por vehículo",
                    )
                    file_summary["visualization_sample_rows"] = int(values.shape[0])
                    file_summary["visualized_features"] = feature_cols
                except Exception as exc:
                    file_summary["feature_visualization_error"] = str(exc)

            summary["files"][alias] = file_summary

        if original_class_counts:
            save_original_class_distribution_plot(
                original_class_counts,
                self._run_figure("eda", "original_temporal_class_distribution.png"),
            )
            original_rows = []
            for source, counts in original_class_counts.items():
                for cls, count in sorted(counts.items()):
                    original_rows.append({"source": source, "class_label": cls, "count": count})
            save_rows_csv(original_rows, self._run_table("eda", "original_temporal_class_distribution.csv"))

        if label_counts:
            save_label_distribution_plot(label_counts, self._run_figure("eda", "binary_label_distribution.png"))
            label_rows = []
            for split, counts in label_counts.items():
                total = sum(counts.values())
                label_rows.append(
                    {
                        "source": split,
                        "normal": counts.get(0, 0),
                        "outlier_or_repair": counts.get(1, 0),
                        "total": total,
                        "positive_ratio": counts.get(1, 0) / total if total else None,
                    }
                )
            save_rows_csv(label_rows, self._run_table("eda", "binary_label_distribution.csv"))

        manifest_path = self.ctx.save_stage_manifest("eda", summary)
        summary["manifest_path"] = str(manifest_path)
        return summary


class PreprocessingWindowingStage(BaseStage):
    """Leakage-safe Spark preprocessing and Spark-native window construction."""

    def _max_vehicles(self) -> int | None:
        execution = self.config.get("execution", {})
        if execution.get("mode", "debug") == "debug":
            return int(execution.get("max_vehicles_debug", 25))
        return None

    def _partitions(self) -> int:
        return int(self.config.get("spark", {}).get("shuffle_partitions", 4))

    def _windows_dir(self, split: str) -> Path:
        return self.ctx.paths.processed_dir / "windows" / split

    def _persist(self, df, name: str):
        cached = df.repartition(self._partitions(), self.vehicle_col).persist(StorageLevel.MEMORY_AND_DISK)
        n_rows = cached.count()
        self.log.info("%s working rows: %s", name, n_rows)
        return cached

    def _debug_vehicle_ids(self, labels_df, max_vehicles: int | None, normal_only: bool = False):
        """Return deterministic debug vehicle IDs.

        Train is sampled only from normal vehicles. Validation/test use a
        deterministic, approximately distribution-preserving binary stratified
        sample with a small minimum number of positives so tiny debug runs still
        exercise PR/F1/ROC evaluation paths.
        """
        if max_vehicles is None:
            return labels_df.select(self.vehicle_col).distinct()

        if normal_only and "in_study_repair" in labels_df.columns:
            return (
                labels_df
                .where(F.col("in_study_repair").cast("int") == 0)
                .select(self.vehicle_col)
                .distinct()
                .orderBy(self.vehicle_col)
                .limit(max_vehicles)
            )

        prepared = prepare_vehicle_level_labels(labels_df, vehicle_col=self.vehicle_col)
        counts = {int(r["y_true"]): int(r["count"]) for r in prepared.groupBy("y_true").count().collect()}
        n_total = counts.get(0, 0) + counts.get(1, 0)
        if n_total == 0:
            return prepared.select(self.vehicle_col).limit(0)

        positive_ratio = counts.get(1, 0) / n_total
        min_positive = int(self.config.get("execution", {}).get("debug_min_positive_vehicles", 2))
        n_positive = max(min_positive, int(round(max_vehicles * positive_ratio))) if counts.get(1, 0) else 0
        n_positive = min(n_positive, counts.get(1, 0), max(0, max_vehicles - 1))
        n_negative = min(max_vehicles - n_positive, counts.get(0, 0))

        positive_ids = (
            prepared.where(F.col("y_true") == 1)
            .select(self.vehicle_col)
            .orderBy(self.vehicle_col)
            .limit(n_positive)
        )
        negative_ids = (
            prepared.where(F.col("y_true") == 0)
            .select(self.vehicle_col)
            .orderBy(self.vehicle_col)
            .limit(n_negative)
        )
        return negative_ids.unionByName(positive_ids).distinct()

    def run(self) -> dict[str, Any]:
        self.log.info("Running preprocessing and optimized Spark-native windowing stage")
        loader = self.loader()
        max_vehicles = self._max_vehicles()

        train_op = loader.read_csv("train_operational")
        validation_op = loader.read_csv("validation_operational")
        test_op = loader.read_csv("test_operational")
        train_tte = loader.read_csv("train_tte")
        validation_labels = loader.read_csv("validation_labels")
        test_labels = loader.read_csv("test_labels")

        if max_vehicles is not None:
            train_ids = self._debug_vehicle_ids(train_tte, max_vehicles, normal_only=True)
            val_ids = self._debug_vehicle_ids(validation_labels, max_vehicles)
            test_ids = self._debug_vehicle_ids(test_labels, max_vehicles)
            train_normal = train_op.join(train_ids, on=self.vehicle_col, how="inner")
            validation_labeled = attach_vehicle_labels(validation_op.join(val_ids, on=self.vehicle_col, how="inner"), validation_labels, vehicle_col=self.vehicle_col)
            test_labeled = attach_vehicle_labels(test_op.join(test_ids, on=self.vehicle_col, how="inner"), test_labels, vehicle_col=self.vehicle_col)
        else:
            train_normal = filter_normal_training_vehicles(train_op, train_tte, vehicle_col=self.vehicle_col)
            validation_labeled = attach_vehicle_labels(validation_op, validation_labels, vehicle_col=self.vehicle_col)
            test_labeled = attach_vehicle_labels(test_op, test_labels, vehicle_col=self.vehicle_col)

        train_normal = self._persist(train_normal, "train")
        validation_labeled = self._persist(validation_labeled, "validation")
        test_labeled = self._persist(test_labeled, "test")

        pre_cfg = self.config.get("preprocessing", {})
        feature_cols = get_numeric_columns(train_normal, exclude=pre_cfg.get("exclude_columns", []))
        missing_report = DataQualityAnalyzer(train_normal).missing_report()
        constant_cols = constant_columns_fast(train_normal, feature_cols) if pre_cfg.get("drop_constant_columns", True) else []

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
            fill_strategy=str(pre_cfg.get("fill_strategy", "median")),
            approximate_quantile_relative_error=float(pre_cfg.get("approximate_quantile_relative_error", 0.05)),
        )

        train_normal.unpersist()
        validation_labeled.unpersist()
        test_labeled.unpersist()

        feature_rows = [
            {"source_feature": src, "scaled_feature": dst}
            for src, dst in zip(metadata.source_feature_cols, metadata.selected_feature_cols)
        ]
        save_rows_csv(feature_rows, self._run_table("preprocessing", "selected_features.csv"))
        dropped_rows = [{"feature": c, "reason": "missing_ratio"} for c in metadata.dropped_missing_cols]
        dropped_rows += [{"feature": c, "reason": "constant"} for c in metadata.dropped_constant_cols]
        save_rows_csv(dropped_rows, self._run_table("preprocessing", "dropped_features.csv"))
        save_rows_csv(
            [
                {
                    "feature": c,
                    "imputation_value": metadata.imputation_values.get(c) if metadata.imputation_values else None,
                    "mean": metadata.scaling_values.get(c, {}).get("mean") if metadata.scaling_values else None,
                    "std": metadata.scaling_values.get(c, {}).get("std") if metadata.scaling_values else None,
                }
                for c in metadata.source_feature_cols
            ],
            self._run_table("preprocessing", "train_fitted_parameters.csv"),
        )
        save_preprocessing_feature_summary(
            len(metadata.source_feature_cols),
            len(metadata.dropped_missing_cols),
            len(metadata.dropped_constant_cols),
            self._run_figure("preprocessing", "feature_selection_summary.png"),
        )

        w_cfg = self.config.get("windowing", {})
        builder = SparkWindowBuilder(
            vehicle_col=self.vehicle_col,
            time_col=self.time_col,
            window_size=int(w_cfg.get("window_size", 30)),
            stride=int(w_cfg.get("stride", 5)),
            label_policy=w_cfg.get("label_policy", "vehicle_label"),
        )
        partitions = int(w_cfg.get("parquet_partitions", self._partitions()))
        split_dfs = {"train": train_proc, "validation": val_proc, "test": test_proc}
        window_paths: dict[str, str] = {}
        split_metadata: dict[str, dict[str, Any]] = {}

        for split, df in split_dfs.items():
            self.log.info("Building %s windows with Spark", split)
            windows_df = builder.build(df, scaled_cols, max_vehicles=None, label_col=self.label_col)
            output_dir = self._windows_dir(split)
            meta = builder.write_parquet(windows_df, output_dir, partitions=partitions)
            window_paths[f"{split}_windows"] = str(output_dir)
            split_metadata[split] = meta.to_dict()

        save_rows_csv(
            [{"split": split, **meta} for split, meta in split_metadata.items()],
            self._run_table("preprocessing", "window_summary_by_split.csv"),
        )
        save_windows_by_split_plot(split_metadata, self._run_figure("preprocessing", "windows_by_split.png"))

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

        # Shared working metadata plus run-scoped evidence.
        active_meta = self.ctx.paths.processed_dir / "metadata" / "preprocessing_metadata.json"
        save_json(metadata_payload, active_meta)
        run_meta = self.ctx.artifact_path("manifests", "preprocessing_metadata.json")
        save_json(metadata_payload, run_meta)
        active_prep = write_active_preparation(
            self.config,
            self.ctx.run_id,
            extra={"window_paths": window_paths, "preprocessing_metadata": str(active_meta)},
        )
        prep_payload = json.loads(active_prep.read_text(encoding="utf-8"))
        save_json(prep_payload, self.ctx.artifact_path("manifests", "preparation.json"))

        output = {"window_paths": window_paths, "metadata_path": str(run_meta), "active_preparation": str(active_prep), **metadata_payload}
        self.ctx.save_stage_manifest("preprocess", output)
        return output


class TrainingStage(BaseStage):
    def _window_path(self, split: str) -> Path:
        return self.ctx.paths.processed_dir / "windows" / split

    def _make_loader(self, split: str, batch_size: int):
        return make_parquet_loader(self._window_path(split), batch_size=batch_size)

    def run(self, model: str = "all") -> dict[str, Any]:
        validate_active_preparation(self.config, self.ctx.run_id)
        self.log.info("Running training stage")
        modeling = self.config.get("modeling", {})
        requested_models = ModelFactory.resolve_requested_models(model, self.config)
        batch_size = int(modeling.get("batch_size", 128))

        train_info = inspect_parquet_windows(self._window_path("train"))
        n_features = int(train_info["n_features"])
        window_size = int(train_info["window_size"])
        if n_features <= 0 or window_size <= 0:
            raise ValueError("Invalid train window metadata. Run --prepare-only first.")

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
            model_path = self.ctx.model_path(model_name, "model.pt")
            trainer.save(model_path)

            validation_rows = score_parquet_windows(
                model_obj,
                self._window_path("validation"),
                batch_size=batch_size,
                device=str(modeling.get("device", "auto")),
            )
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
            threshold_path = self.ctx.model_path(model_name, "threshold.json")
            history_path = self.ctx.model_path(model_name, "training_history.json")
            save_json(threshold_payload, threshold_path)
            save_json({"model": model_name, "history": history}, history_path)
            save_training_history_plot(
                history,
                self._run_figure("training", f"{model_name}_loss_curve.png"),
                model_name,
            )

            out["models"][model_name] = {
                "model_path": str(model_path),
                "threshold_path": str(threshold_path),
                "training_history_path": str(history_path),
                "threshold": float(threshold),
                "training_time_seconds": float(history.get("training_time_seconds", 0.0)),
            }

        self.ctx.save_stage_manifest("train", out)
        return out


class EvaluationStage(BaseStage):
    def _window_path(self, split: str) -> Path:
        return self.ctx.paths.processed_dir / "windows" / split

    def _load_threshold(self, model_name: str) -> dict[str, Any]:
        path = self.ctx.model_path(model_name, "threshold.json")
        if not path.exists():
            raise FileNotFoundError(f"Threshold file not found for {model_name}: {path}")
        return json.loads(path.read_text(encoding="utf-8"))

    def _load_model(self, model_name: str, n_features: int, window_size: int):
        model = ModelFactory.create(model_name, n_features=n_features, window_size=window_size, config=self.config)
        model_path = self.ctx.model_path(model_name, "model.pt")
        if not model_path.exists():
            raise FileNotFoundError(f"Model not found: {model_path}. Train this model first for run {self.ctx.run_id}.")
        state = torch.load(model_path, map_location="cpu")
        model.load_state_dict(state)
        return model

    def run(self, model: str = "all", split: str | None = None) -> dict[str, Any]:
        validate_active_preparation(self.config, self.ctx.run_id)
        self.log.info("Running evaluation stage")
        requested_models = ModelFactory.resolve_requested_models(model, self.config)
        eval_cfg = self.config.get("evaluation", {})
        split = split or eval_cfg.get("primary_split", "test")
        batch_size = int(self.config.get("modeling", {}).get("batch_size", 128))
        device = str(self.config.get("modeling", {}).get("device", "auto"))

        window_info = inspect_parquet_windows(self._window_path(split))
        n_features = int(window_info["n_features"])
        window_size = int(window_info["window_size"])
        output: dict[str, Any] = {"split": split, "models": {}}

        for model_name in requested_models:
            threshold_payload = self._load_threshold(model_name)
            threshold = float(threshold_payload["threshold"])
            vehicle_score_col = threshold_payload.get("score_col", "max_score")
            training_time = float(threshold_payload.get("history", {}).get("training_time_seconds", 0.0))
            model_obj = self._load_model(model_name, n_features=n_features, window_size=window_size)

            start = time.time()
            window_rows = score_parquet_windows(model_obj, self._window_path(split), batch_size=batch_size, device=device)
            inference_time = round(time.time() - start, 3)
            window_rows = threshold_rows(window_rows, threshold, score_col="outlier_score", pred_col="is_outlier")
            vehicle_rows = classify_vehicle_scores(
                aggregate_vehicle_scores(window_rows),
                threshold,
                score_col=vehicle_score_col,
            )

            vehicle_metrics = binary_classification_metrics(
                [r["y_true"] for r in vehicle_rows],
                [r["is_outlier"] for r in vehicle_rows],
                scores=[r[vehicle_score_col] for r in vehicle_rows],
            )
            window_scores = np.asarray([r["outlier_score"] for r in window_rows], dtype=float)
            window_score_summary = {
                "model": model_name,
                "split": split,
                "level": "window_descriptive",
                "n_windows": int(len(window_rows)),
                "n_vehicles": int(len({r["vehicle_id"] for r in window_rows})),
                "mean_outlier_score": float(np.mean(window_scores)) if window_scores.size else None,
                "median_outlier_score": float(np.median(window_scores)) if window_scores.size else None,
                "p95_outlier_score": float(np.percentile(window_scores, 95)) if window_scores.size else None,
                "note": (
                    "No se calculan Precision/Recall/F1 por ventana porque SCANIA Component X no proporciona "
                    "ground truth de anomalía para cada ventana; la referencia supervisada se aplica a nivel vehículo."
                ),
            }
            common = {
                "model": model_name,
                "split": split,
                "threshold": threshold,
                "training_time_seconds": training_time,
                "inference_time_seconds": inference_time,
            }
            vehicle_metrics.update({**common, "level": "vehicle"})

            save_predictions_table(window_rows, self.ctx.artifact_path("predictions", model_name, f"{split}_window_predictions.csv"))
            save_predictions_table(vehicle_rows, self.ctx.artifact_path("predictions", model_name, f"{split}_vehicle_predictions.csv"))
            save_json(vehicle_metrics, self.ctx.artifact_path("metrics", f"{model_name}_{split}_vehicle_metrics.json"))
            save_json(window_score_summary, self.ctx.artifact_path("metrics", f"{model_name}_{split}_window_score_summary.json"))

            y_true = [r["y_true"] for r in vehicle_rows]
            y_pred = [r["is_outlier"] for r in vehicle_rows]
            scores = [r[vehicle_score_col] for r in vehicle_rows]
            eval_fig_dir = "evaluation"
            save_confusion_matrix_plot(y_true, y_pred, self._run_figure(eval_fig_dir, f"{model_name}_confusion_matrix.png"), model_name)
            save_precision_recall_curve(y_true, scores, self._run_figure(eval_fig_dir, f"{model_name}_precision_recall_curve.png"), model_name)
            save_roc_curve(y_true, scores, self._run_figure(eval_fig_dir, f"{model_name}_roc_curve.png"), model_name)
            save_score_distribution(y_true, scores, self._run_figure(eval_fig_dir, f"{model_name}_score_distribution.png"), model_name)
            save_score_boxplot(y_true, scores, self._run_figure(eval_fig_dir, f"{model_name}_score_boxplot.png"), model_name)

            output["models"][model_name] = {"vehicle_metrics": vehicle_metrics, "window_score_summary": window_score_summary}

        self.ctx.save_stage_manifest("evaluate", output)
        return output


class ComparisonStage(BaseStage):
    """Consolidate current run metrics; no other run can leak into this table."""

    def run(self) -> dict[str, Any]:
        files = sorted(self.ctx.paths.metrics_dir.glob("*_metrics.json"))
        rows: list[dict[str, Any]] = []
        for path in files:
            rows.append(json.loads(path.read_text(encoding="utf-8")))
        if not rows:
            raise FileNotFoundError(
                f"No hay métricas para comparar en {self.ctx.paths.metrics_dir}. Evalúe primero los tres modelos para este run."
            )

        expected_models = set(self.config.get("modeling", {}).get("models", []))
        available_vehicle_models = {str(r.get("model")) for r in rows if r.get("level") == "vehicle"}
        missing_models = sorted(expected_models - available_vehicle_models)
        if missing_models and not bool(self.config.get("comparison", {}).get("allow_partial", False)):
            raise RuntimeError(
                "La comparación requiere los tres modelos del experimento. Faltan métricas a nivel vehículo para: "
                + ", ".join(missing_models)
            )

        out_path = self.ctx.artifact_path("comparisons", "comparison_all_models.csv")
        save_rows_csv(rows, out_path)
        save_model_comparison_plot(rows, self._run_figure("comparison", "model_metrics_comparison.png"), level="vehicle")
        save_runtime_comparison_plot(rows, self._run_figure("comparison", "model_runtime_comparison.png"), level="vehicle")

        vehicle_rows = [r for r in rows if r.get("level") == "vehicle"]
        best_pr = max(vehicle_rows, key=lambda r: float(r.get("pr_auc") if r.get("pr_auc") is not None else -1), default=None)
        best_f1 = max(vehicle_rows, key=lambda r: float(r.get("f1_score") if r.get("f1_score") is not None else -1), default=None)
        summary = {
            "comparison_path": str(out_path),
            "n_rows": len(rows),
            "models": sorted({str(r.get("model")) for r in rows}),
            "best_vehicle_pr_auc": best_pr,
            "best_vehicle_f1": best_f1,
        }
        save_json(summary, self.ctx.artifact_path("comparisons", "comparison_summary.json"))
        self.ctx.save_stage_manifest("compare", summary)
        return summary


class ReportingStage(BaseStage):
    """Create a compact inventory and run summary from already generated artifacts."""

    def run(self) -> dict[str, Any]:
        run_dir = self.ctx.paths.run_dir
        inventory: list[dict[str, Any]] = []
        for path in sorted(run_dir.rglob("*")):
            if path.is_file():
                inventory.append(
                    {
                        "relative_path": str(path.relative_to(run_dir)),
                        "size_bytes": path.stat().st_size,
                        "suffix": path.suffix.lower(),
                    }
                )
        inventory_path = self._run_table("report", "artifact_inventory.csv")
        save_rows_csv(inventory, inventory_path)

        metric_files = sorted(self.ctx.paths.metrics_dir.glob("*_metrics.json"))
        metrics = [json.loads(p.read_text(encoding="utf-8")) for p in metric_files]
        vehicle_metrics = [m for m in metrics if m.get("level") == "vehicle"]
        best_pr = max(vehicle_metrics, key=lambda r: float(r.get("pr_auc") if r.get("pr_auc") is not None else -1), default=None)

        figure_counts: dict[str, int] = {}
        figures_root = self.ctx.paths.figures_dir
        if figures_root.exists():
            for folder in ["eda", "preprocessing", "training", "evaluation", "comparison"]:
                figure_counts[folder] = len(list((figures_root / folder).glob("*.png"))) if (figures_root / folder).exists() else 0

        summary = {
            "run_name": self.ctx.run_id,
            "run_dir": str(run_dir),
            "n_artifacts": len(inventory),
            "n_metric_files": len(metric_files),
            "figure_counts": figure_counts,
            "best_vehicle_model_by_pr_auc": best_pr,
            "notes": [
                "Las figuras de EDA basadas en variables operacionales usan una muestra acotada para visualización; las métricas de evaluación no se calculan sobre esa muestra.",
                "La comparación final usa exclusivamente los artefactos del run actual.",
            ],
        }
        save_json(summary, self.ctx.artifact_path("manifests", "report.json"))
        return summary

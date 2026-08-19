from __future__ import annotations

from typing import Iterable, Optional

try:
    from pyspark.sql import DataFrame
    from pyspark.sql import functions as F
except Exception:  # pragma: no cover - allows local tests without PySpark
    DataFrame = object  # type: ignore
    F = None  # type: ignore


DEFAULT_LABEL_CANDIDATES = (
    "class_label",
    "in_study_repair",
    "is_outlier",
    "outlier",
    "label",
    "target",
    "is_anomaly",
    "anomaly",
    "repair",
    "failure",
)


def to_binary_reference_label(value: int | float | None) -> int | None:
    """Collapse the official temporal class into the binary TFM reference target.

    ``0`` remains negative/reference; any positive class (1..4 in the official
    validation/test labels) becomes positive. ``None`` remains missing.
    """
    if value is None:
        return None
    return 1 if int(value) > 0 else 0


def infer_label_column(df: DataFrame, candidates: Iterable[str] = DEFAULT_LABEL_CANDIDATES) -> str:
    """Infer the available reference-label column from common names.

    The SCANIA Component X validation/test label files use `class_label`. Earlier
    prototypes did not include this name, which could break label attachment.
    """
    available = set(df.columns)
    for candidate in candidates:
        if candidate in available:
            return candidate
    raise ValueError(f"No label column found. Checked candidates: {list(candidates)}")


def prepare_vehicle_level_labels(
    labels_df: DataFrame,
    vehicle_col: str = "vehicle_id",
    label_col: Optional[str] = None,
    output_col: str = "y_true",
) -> DataFrame:
    """Create one binary label per vehicle from validation/test labels or train_tte.

    Validation/test expose five temporal classes (0..4). For the binary outlier
    task of this project, class 0 is used as the negative/reference group and
    classes 1..4 are collapsed into a positive group. This conversion is an
    operational evaluation reference, not a claim that the dataset provides
    intrinsic point-wise anomaly labels. If a vehicle has multiple rows, it is
    considered positive when at least one row is positive.
    """
    label_col = label_col or infer_label_column(labels_df)
    raw_label = F.col(label_col).cast("int")

    # The official validation/test files use five temporal classes (0..4).
    # For this TFM the evaluation target is binary outlier detection: class 0
    # is the negative/reference group (>48 time steps before failure), while
    # classes 1..4 are collapsed into the positive group (within 48 time steps).
    # train_tte already uses 0/1 and is therefore preserved by the same rule.
    binary_label = (
        F.when(raw_label.isNull(), F.lit(None).cast("int"))
        .when(raw_label > 0, F.lit(1))
        .otherwise(F.lit(0))
        .alias(output_col)
    )
    return (
        labels_df
        .select(vehicle_col, binary_label)
        .groupBy(vehicle_col)
        .agg(F.max(output_col).alias(output_col))
    )


def attach_vehicle_labels(
    operational_df: DataFrame,
    labels_df: DataFrame,
    vehicle_col: str = "vehicle_id",
    label_col: Optional[str] = None,
    output_col: str = "y_true",
    fill_unlabeled_with: int = -1,
) -> DataFrame:
    """Join operational readouts with vehicle-level labels.

    Vehicles without a reference label are marked as -1 by default. They can be
    kept for scoring but are ignored by supervised metrics.
    """
    vehicle_labels = prepare_vehicle_level_labels(labels_df, vehicle_col, label_col, output_col)
    return (
        operational_df
        .join(vehicle_labels, on=vehicle_col, how="left")
        .withColumn(output_col, F.coalesce(F.col(output_col), F.lit(fill_unlabeled_with)).cast("int"))
    )


def filter_normal_training_vehicles(
    operational_df: DataFrame,
    train_tte_df: DataFrame,
    vehicle_col: str = "vehicle_id",
    repair_col: str = "in_study_repair",
) -> DataFrame:
    """Keep vehicles without in-study repair for normality-oriented training."""
    normal_vehicles = (
        train_tte_df
        .where(F.col(repair_col).cast("int") == 0)
        .select(vehicle_col)
        .distinct()
    )
    return operational_df.join(normal_vehicles, on=vehicle_col, how="inner")


def summarize_vehicle_labels(labels_df: DataFrame, vehicle_col: str = "vehicle_id", label_col: Optional[str] = None) -> dict:
    """Return a compact label summary useful for EDA tables."""
    label_col = label_col or infer_label_column(labels_df)
    prepared = prepare_vehicle_level_labels(labels_df, vehicle_col, label_col)
    rows = prepared.groupBy("y_true").count().collect()
    counts = {int(r["y_true"]): int(r["count"]) for r in rows}
    total = sum(counts.values())
    return {
        "total_labeled_vehicles": total,
        "normal_vehicles": counts.get(0, 0),
        "outlier_or_repair_vehicles": counts.get(1, 0),
        "positive_ratio": counts.get(1, 0) / total if total else None,
        "label_column": label_col,
    }

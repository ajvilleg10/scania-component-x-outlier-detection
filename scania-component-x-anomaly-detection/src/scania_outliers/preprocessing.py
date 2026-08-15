from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, List

from pyspark.sql import DataFrame
from pyspark.sql import functions as F


@dataclass
class PreprocessingMetadata:
    selected_feature_cols: list[str]
    source_feature_cols: list[str]
    dropped_missing_cols: list[str]
    dropped_constant_cols: list[str]
    max_missing_ratio: float
    fill_strategy: str
    scaler_fit_split: str = "train"
    scaling_method: str = "scalar_standard"
    imputation_values: dict[str, float] | None = None
    scaling_values: dict[str, dict[str, float]] | None = None

    def to_dict(self) -> dict:
        return asdict(self)


def get_numeric_columns(df: DataFrame, exclude: list[str] | None = None) -> List[str]:
    """Return numeric columns excluding identifiers and target-like fields."""
    exclude = set(exclude or [])
    numeric_types = {"int", "bigint", "double", "float", "long", "short", "decimal"}
    return [c for c, t in df.dtypes if any(t.startswith(nt) for nt in numeric_types) and c not in exclude]


def _safe_numeric(col_name: str, fill_value: float | None = None):
    """Numeric Spark expression that treats null and NaN as missing."""
    value = F.col(col_name).cast("double")
    if fill_value is None:
        return F.when(F.col(col_name).isNull() | F.isnan(value), None).otherwise(value)
    return F.when(F.col(col_name).isNull() | F.isnan(value), F.lit(float(fill_value))).otherwise(value)


def columns_above_missing_ratio(missing_report, max_missing_ratio: float = 0.8) -> list[str]:
    return [
        str(row["column"])
        for row in missing_report
        if float(row.get("missing_ratio", 0.0)) > max_missing_ratio
    ]


def drop_columns_by_missing_ratio(df: DataFrame, missing_report, max_missing_ratio: float = 0.8) -> tuple[DataFrame, list[str]]:
    cols_to_drop = columns_above_missing_ratio(missing_report, max_missing_ratio)
    return df.drop(*cols_to_drop), cols_to_drop


def constant_columns_fast(df: DataFrame, columns: list[str]) -> list[str]:
    """Detect constant columns in a single Spark aggregation.

    The previous implementation executed one distinct/count job per column. With 100+
    SCANIA variables this was slow in Colab. This version computes min/max for all
    candidate numeric columns in one pass and marks a column as constant when min=max.
    """
    if not columns:
        return []
    exprs = []
    for c in columns:
        v = _safe_numeric(c)
        exprs.append(F.min(v).alias(f"{c}__min"))
        exprs.append(F.max(v).alias(f"{c}__max"))
    row = df.select(*exprs).collect()[0].asDict()
    constant: list[str] = []
    for c in columns:
        mn = row.get(f"{c}__min")
        mx = row.get(f"{c}__max")
        if mn is None and mx is None:
            constant.append(c)
        elif mn == mx:
            constant.append(c)
    return constant


def fit_imputation_values(df: DataFrame, input_cols: list[str], strategy: str = "median", relative_error: float = 0.05) -> dict[str, float]:
    """Fit imputation values on train only without Spark ML Imputer.

    `approxQuantile` accepts a list of columns and avoids creating a large ML vector.
    For Colab this is much lighter than fitting `Imputer` + `VectorAssembler` over
    the full wide dataframe.
    """
    if not input_cols:
        return {}
    strategy = strategy.lower()
    if strategy == "median":
        quantiles = df.approxQuantile(input_cols, [0.5], float(relative_error))
        values = {}
        for c, qs in zip(input_cols, quantiles):
            values[c] = float(qs[0]) if qs else 0.0
        return values

    if strategy in {"mean", "average"}:
        exprs = [F.mean(_safe_numeric(c)).alias(c) for c in input_cols]
        row = df.select(*exprs).collect()[0].asDict()
        return {c: float(row.get(c) or 0.0) for c in input_cols}

    if strategy in {"zero", "constant"}:
        return {c: 0.0 for c in input_cols}

    raise ValueError(f"Unsupported fill_strategy: {strategy}")


def fit_scaling_values(df: DataFrame, input_cols: list[str], fill_values: dict[str, float]) -> dict[str, dict[str, float]]:
    """Fit mean/std on train only using scalar expressions."""
    if not input_cols:
        return {}
    exprs = []
    for c in input_cols:
        filled = _safe_numeric(c, fill_values.get(c, 0.0))
        exprs.append(F.mean(filled).alias(f"{c}__mean"))
        exprs.append(F.stddev_pop(filled).alias(f"{c}__std"))
    row = df.select(*exprs).collect()[0].asDict()
    stats: dict[str, dict[str, float]] = {}
    for c in input_cols:
        mean = float(row.get(f"{c}__mean") or 0.0)
        std = float(row.get(f"{c}__std") or 0.0)
        if std == 0.0:
            std = 1.0
        stats[c] = {"mean": mean, "std": std}
    return stats


def apply_scalar_preprocessing(
    df: DataFrame,
    input_cols: list[str],
    fill_values: dict[str, float],
    scaling_values: dict[str, dict[str, float]],
    vehicle_col: str = "vehicle_id",
    time_col: str = "time_step",
    label_col: str = "y_true",
    output_prefix: str = "x",
) -> tuple[DataFrame, list[str]]:
    """Apply imputation and standard scaling using one Spark select.

    This avoids VectorAssembler, StandardScaler and vector_to_array, which created a
    very large Spark plan and caused the Java gateway to die in Colab.
    """
    out_cols = [F.col(vehicle_col), F.col(time_col)]
    if label_col in df.columns:
        out_cols.append(F.col(label_col).cast("int").alias(label_col))

    scaled_cols: list[str] = []
    for i, c in enumerate(input_cols):
        out_name = f"{output_prefix}_{i}"
        mean = scaling_values.get(c, {}).get("mean", 0.0)
        std = scaling_values.get(c, {}).get("std", 1.0) or 1.0
        filled = _safe_numeric(c, fill_values.get(c, 0.0))
        scaled = ((filled - F.lit(float(mean))) / F.lit(float(std))).cast("float").alias(out_name)
        out_cols.append(scaled)
        scaled_cols.append(out_name)

    return df.select(*out_cols), scaled_cols


def apply_train_fitted_preprocessing(
    train_df: DataFrame,
    validation_df: DataFrame,
    test_df: DataFrame,
    feature_cols: list[str],
    missing_report,
    constant_cols: list[str] | None = None,
    max_missing_ratio: float = 0.8,
    vehicle_col: str = "vehicle_id",
    time_col: str = "time_step",
    label_col: str = "y_true",
    fill_strategy: str = "median",
    approximate_quantile_relative_error: float = 0.05,
) -> tuple[DataFrame, DataFrame, DataFrame, list[str], PreprocessingMetadata]:
    """Leakage-safe scalar preprocessing fitted only on train.

    Optimized for Google Colab:
    - no Pandas/toPandas;
    - no Spark ML StandardScaler/VectorAssembler;
    - no iterative withColumn over 100+ features;
    - all imputation/scaling parameters are fitted only on train.
    """
    constant_cols = constant_cols or []
    missing_cols = [c for c in columns_above_missing_ratio(missing_report, max_missing_ratio) if c in feature_cols]
    drop_cols = sorted(set(missing_cols + [c for c in constant_cols if c in feature_cols]))
    selected = [c for c in feature_cols if c not in drop_cols]

    train_base = train_df.drop(*drop_cols) if drop_cols else train_df
    val_base = validation_df.drop(*drop_cols) if drop_cols else validation_df
    test_base = test_df.drop(*drop_cols) if drop_cols else test_df

    fill_values = fit_imputation_values(
        train_base,
        selected,
        strategy=fill_strategy,
        relative_error=approximate_quantile_relative_error,
    )
    scaling_values = fit_scaling_values(train_base, selected, fill_values)

    train_out, scaled_cols = apply_scalar_preprocessing(
        train_base, selected, fill_values, scaling_values, vehicle_col=vehicle_col, time_col=time_col, label_col=label_col
    )
    val_out, _ = apply_scalar_preprocessing(
        val_base, selected, fill_values, scaling_values, vehicle_col=vehicle_col, time_col=time_col, label_col=label_col
    )
    test_out, _ = apply_scalar_preprocessing(
        test_base, selected, fill_values, scaling_values, vehicle_col=vehicle_col, time_col=time_col, label_col=label_col
    )

    metadata = PreprocessingMetadata(
        selected_feature_cols=scaled_cols,
        source_feature_cols=selected,
        dropped_missing_cols=missing_cols,
        dropped_constant_cols=[c for c in constant_cols if c in feature_cols],
        max_missing_ratio=max_missing_ratio,
        fill_strategy=fill_strategy,
        imputation_values=fill_values,
        scaling_values=scaling_values,
    )
    return train_out, val_out, test_out, scaled_cols, metadata

from __future__ import annotations

try:
    from pyspark.sql import DataFrame, Window
    from pyspark.sql import functions as F
except Exception:  # pragma: no cover
    DataFrame = object  # type: ignore


def time_step_gap_report(
    df: DataFrame,
    vehicle_col: str = "vehicle_id",
    time_col: str = "time_step",
    sample_fraction: float | None = None,
) -> list[dict]:
    """Summarize gaps between consecutive time steps by vehicle using Spark only."""
    work = df
    if sample_fraction is not None and 0 < sample_fraction < 1:
        work = work.sample(withReplacement=False, fraction=sample_fraction, seed=42)

    w = Window.partitionBy(vehicle_col).orderBy(time_col)
    gaps = (
        work
        .select(vehicle_col, F.col(time_col).cast("double").alias(time_col))
        .withColumn("prev_time_step", F.lag(time_col).over(w))
        .withColumn("delta_time_step", F.col(time_col) - F.col("prev_time_step"))
        .where(F.col("delta_time_step").isNotNull())
    )
    row = gaps.select(
        F.count("delta_time_step").alias("n_gaps"),
        F.mean("delta_time_step").alias("mean_gap"),
        F.expr("percentile_approx(delta_time_step, 0.5)").alias("median_gap"),
        F.expr("percentile_approx(delta_time_step, 0.95)").alias("p95_gap"),
        F.min("delta_time_step").alias("min_gap"),
        F.max("delta_time_step").alias("max_gap"),
    ).collect()[0].asDict()
    return [{k: (float(v) if isinstance(v, float) else int(v) if isinstance(v, int) else v) for k, v in row.items()}]


def trajectory_length_report(df: DataFrame, vehicle_col: str = "vehicle_id") -> list[dict]:
    """Return summary statistics of number of records per vehicle using Spark only."""
    counts = df.groupBy(vehicle_col).count()
    row = counts.select(
        F.count("count").alias("n_vehicles"),
        F.mean("count").alias("mean_records"),
        F.expr("percentile_approx(count, 0.5)").alias("median_records"),
        F.expr("percentile_approx(count, 0.05)").alias("p05_records"),
        F.expr("percentile_approx(count, 0.95)").alias("p95_records"),
        F.min("count").alias("min_records"),
        F.max("count").alias("max_records"),
    ).collect()[0].asDict()
    return [{k: (float(v) if isinstance(v, float) else int(v) if isinstance(v, int) else v) for k, v in row.items()}]

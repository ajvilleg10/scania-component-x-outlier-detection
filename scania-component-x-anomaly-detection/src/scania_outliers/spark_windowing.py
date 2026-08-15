from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Literal

try:
    from pyspark.sql import DataFrame
    from pyspark.sql import functions as F
except Exception:  # pragma: no cover
    DataFrame = object  # type: ignore

LabelPolicy = Literal["vehicle_label", "max_label_in_window", "unlabeled"]


@dataclass(frozen=True)
class SparkWindowMetadata:
    n_windows: int
    n_vehicles: int
    window_size: int
    stride: int
    n_features: int
    output_path: str

    def to_dict(self) -> dict:
        return {
            "n_windows": int(self.n_windows),
            "n_vehicles": int(self.n_vehicles),
            "window_size": int(self.window_size),
            "stride": int(self.stride),
            "n_features": int(self.n_features),
            "output_path": self.output_path,
        }


class SparkWindowBuilder:
    """Build fixed-length multivariate windows directly with Spark.

    This builder avoids DataFrame.toPandas(). Windows are persisted as Parquet,
    which is safer for Colab and scalable enough for the SCANIA Component X files.
    """

    def __init__(
        self,
        vehicle_col: str = "vehicle_id",
        time_col: str = "time_step",
        window_size: int = 30,
        stride: int = 5,
        label_policy: LabelPolicy = "vehicle_label",
    ):
        self.vehicle_col = vehicle_col
        self.time_col = time_col
        self.window_size = int(window_size)
        self.stride = int(stride)
        self.label_policy = label_policy

    def _filter_debug_vehicles(self, df: DataFrame, max_vehicles: int | None) -> DataFrame:
        if max_vehicles is None:
            return df
        vehicles = df.select(self.vehicle_col).distinct().limit(int(max_vehicles))
        return df.join(vehicles, on=self.vehicle_col, how="inner")

    def build(self, df: DataFrame, feature_cols: list[str], max_vehicles: int | None = None, label_col: str | None = None) -> DataFrame:
        if not feature_cols:
            raise ValueError("feature_cols cannot be empty when building Spark windows")

        work = self._filter_debug_vehicles(df, max_vehicles)
        feature_array = F.array(*[F.col(c).cast("float") for c in feature_cols])

        selected = [
            F.col(self.vehicle_col),
            F.col(self.time_col),
            feature_array.alias("features"),
        ]
        use_label = bool(label_col and label_col in work.columns and self.label_policy != "unlabeled")
        if use_label:
            selected.append(F.col(label_col).cast("int").alias("row_label"))

        work = work.select(*selected)

        if use_label:
            seq_item = F.struct(
                F.col(self.time_col).alias("time_step"),
                F.col("features").alias("features"),
                F.col("row_label").alias("label"),
            )
        else:
            seq_item = F.struct(
                F.col(self.time_col).alias("time_step"),
                F.col("features").alias("features"),
            )

        grouped = (
            work
            .groupBy(self.vehicle_col)
            .agg(F.sort_array(F.collect_list(seq_item)).alias("sequence"))
            .withColumn("sequence_length", F.size("sequence"))
            .where(F.col("sequence_length") >= F.lit(self.window_size))
        )

        starts = grouped.withColumn(
            "start_idx",
            F.explode(F.sequence(F.lit(0), F.col("sequence_length") - F.lit(self.window_size), F.lit(self.stride))),
        )

        windows = (
            starts
            .withColumn("window", F.slice(F.col("sequence"), F.col("start_idx") + F.lit(1), self.window_size))
            .withColumn("X", F.expr("transform(window, x -> x.features)"))
            .withColumn("start_time_step", F.expr("element_at(window, 1).time_step"))
            .withColumn("end_time_step", F.expr(f"element_at(window, {self.window_size}).time_step"))
            .withColumn("window_id", F.concat_ws("_", F.col(self.vehicle_col).cast("string"), F.col("start_idx").cast("string")))
        )

        if use_label:
            if self.label_policy == "vehicle_label":
                # Vehicle-level labels are repeated across rows after joining; max keeps a single vehicle reference.
                label_expr = "aggregate(transform(window, x -> coalesce(x.label, -1)), -1, (acc, x) -> greatest(acc, x))"
            elif self.label_policy == "max_label_in_window":
                label_expr = "aggregate(transform(window, x -> coalesce(x.label, -1)), -1, (acc, x) -> greatest(acc, x))"
            else:
                label_expr = "-1"
            windows = windows.withColumn("y_true", F.expr(label_expr).cast("int"))
        else:
            windows = windows.withColumn("y_true", F.lit(-1).cast("int"))

        return windows.select(
            "window_id",
            F.col(self.vehicle_col).alias("vehicle_id"),
            "start_idx",
            "start_time_step",
            "end_time_step",
            "X",
            "y_true",
        )

    def write_parquet(self, windows_df: DataFrame, output_path: str | Path, partitions: int = 8) -> SparkWindowMetadata:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        n_windows = windows_df.count()
        n_vehicles = windows_df.select("vehicle_id").distinct().count()
        first = windows_df.select("X").head()
        n_features = 0
        if first is not None and first["X"]:
            n_features = len(first["X"][0])
        (
            windows_df
            .repartition(int(partitions))
            .write
            .mode("overwrite")
            .parquet(str(output_path))
        )
        return SparkWindowMetadata(
            n_windows=n_windows,
            n_vehicles=n_vehicles,
            window_size=self.window_size,
            stride=self.stride,
            n_features=n_features,
            output_path=str(output_path),
        )

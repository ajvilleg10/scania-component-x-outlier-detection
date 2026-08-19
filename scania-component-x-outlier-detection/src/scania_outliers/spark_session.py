from __future__ import annotations


class SparkSessionFactory:
    """Factory responsible for creating Spark sessions in Colab/local environments.

    The pipeline avoids Pandas/toPandas in the main path. Arrow is disabled through
    the current PySpark config only (`spark.sql.execution.arrow.pyspark.enabled`).
    The deprecated `spark.sql.execution.arrow.enabled` is intentionally not set so
    Colab logs do not get flooded with deprecation warnings.
    """

    @staticmethod
    def create(
        app_name: str = "ScaniaComponentXOutlierDetection",
        master: str = "local[*]",
        shuffle_partitions: int = 4,
        driver_memory: str = "8g",
        arrow_enabled: bool = False,
        max_to_string_fields: int = 400,
        adaptive_enabled: bool = True,
    ):
        try:
            from pyspark.sql import SparkSession
        except ModuleNotFoundError as exc:  # pragma: no cover
            raise ModuleNotFoundError(
                "PySpark is required for data processing stages. Install requirements.txt before running the pipeline."
            ) from exc

        spark = (
            SparkSession.builder
            .appName(app_name)
            .master(master)
            .config("spark.sql.shuffle.partitions", str(shuffle_partitions))
            .config("spark.default.parallelism", str(shuffle_partitions))
            .config("spark.driver.memory", driver_memory)
            .config("spark.sql.debug.maxToStringFields", str(max_to_string_fields))
            .config("spark.sql.execution.arrow.pyspark.enabled", str(arrow_enabled).lower())
            .config("spark.sql.execution.arrow.pyspark.fallback.enabled", "true")
            .config("spark.sql.adaptive.enabled", str(adaptive_enabled).lower())
            .config("spark.sql.adaptive.coalescePartitions.enabled", str(adaptive_enabled).lower())
            .config("spark.serializer", "org.apache.spark.serializer.KryoSerializer")
            .config("spark.ui.showConsoleProgress", "false")
            .getOrCreate()
        )
        spark.sparkContext.setLogLevel("ERROR")
        return spark


def create_spark_session(config: dict | None = None):
    """Create Spark session from project configuration."""
    cfg = (config or {}).get("spark", {})
    return SparkSessionFactory.create(
        app_name=cfg.get("app_name", "ScaniaComponentXOutlierDetection"),
        master=cfg.get("master", "local[*]"),
        shuffle_partitions=int(cfg.get("shuffle_partitions", 4)),
        driver_memory=cfg.get("driver_memory", "8g"),
        arrow_enabled=bool(cfg.get("arrow_enabled", False)),
        max_to_string_fields=int(cfg.get("debug_max_to_string_fields", 400)),
        adaptive_enabled=bool(cfg.get("adaptive_enabled", True)),
    )

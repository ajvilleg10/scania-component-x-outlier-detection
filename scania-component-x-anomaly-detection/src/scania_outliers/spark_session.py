from __future__ import annotations


class SparkSessionFactory:
    """Factory responsible for creating Spark sessions in Colab/local environments.

    Arrow is disabled by default because the project avoids Pandas/toPandas in the
    main pipeline. This prevents Java/Arrow compatibility errors commonly seen in
    Google Colab when Spark tries to transfer large DataFrames to the Python driver.
    """

    @staticmethod
    def create(
        app_name: str = "ScaniaComponentXOutlierDetection",
        master: str = "local[*]",
        shuffle_partitions: int = 8,
        driver_memory: str = "12g",
        arrow_enabled: bool = False,
        max_to_string_fields: int = 200,
    ):
        try:
            from pyspark.sql import SparkSession
        except ModuleNotFoundError as exc:  # pragma: no cover
            raise ModuleNotFoundError(
                "PySpark is required for data processing stages. Install the project with `pip install -e .` "
                "or install pyspark before running EDA/preprocessing."
            ) from exc

        spark = (
            SparkSession.builder
            .appName(app_name)
            .master(master)
            .config("spark.sql.shuffle.partitions", str(shuffle_partitions))
            .config("spark.driver.memory", driver_memory)
            .config("spark.sql.debug.maxToStringFields", str(max_to_string_fields))
            .config("spark.sql.execution.arrow.pyspark.enabled", str(arrow_enabled).lower())
            .config("spark.sql.execution.arrow.enabled", str(arrow_enabled).lower())
            .config("spark.sql.execution.arrow.pyspark.fallback.enabled", "true")
            .getOrCreate()
        )
        spark.sparkContext.setLogLevel("WARN")
        return spark


def create_spark_session(config: dict | None = None):
    """Create Spark session from project configuration."""
    cfg = (config or {}).get("spark", {})
    return SparkSessionFactory.create(
        app_name=cfg.get("app_name", "ScaniaComponentXOutlierDetection"),
        master=cfg.get("master", "local[*]"),
        shuffle_partitions=int(cfg.get("shuffle_partitions", 8)),
        driver_memory=cfg.get("driver_memory", "12g"),
        arrow_enabled=bool(cfg.get("arrow_enabled", False)),
        max_to_string_fields=int(cfg.get("debug_max_to_string_fields", 200)),
    )

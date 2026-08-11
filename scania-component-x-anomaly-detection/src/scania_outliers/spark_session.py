from __future__ import annotations


class SparkSessionFactory:
    """Factory responsible for creating Spark sessions in Colab/local environments."""

    @staticmethod
    def create(
        app_name: str = "ScaniaComponentXOutlierDetection",
        master: str = "local[*]",
        shuffle_partitions: int = 64,
        driver_memory: str = "8g",
        arrow_enabled: bool = True,
    ):
        try:
            from pyspark.sql import SparkSession
        except ModuleNotFoundError as exc:  # pragma: no cover
            raise ModuleNotFoundError(
                "PySpark is required for data processing stages. Install the project with `pip install -e .` "
                "or install pyspark before running EDA/preprocessing."
            ) from exc
        return (
            SparkSession.builder
            .appName(app_name)
            .master(master)
            .config("spark.sql.shuffle.partitions", str(shuffle_partitions))
            .config("spark.driver.memory", driver_memory)
            .config("spark.sql.execution.arrow.pyspark.enabled", str(arrow_enabled).lower())
            .getOrCreate()
        )


def create_spark_session(config: dict | None = None):
    """Create Spark session from project configuration."""
    cfg = (config or {}).get("spark", {})
    return SparkSessionFactory.create(
        app_name=cfg.get("app_name", "ScaniaComponentXOutlierDetection"),
        master=cfg.get("master", "local[*]"),
        shuffle_partitions=int(cfg.get("shuffle_partitions", 64)),
        driver_memory=cfg.get("driver_memory", "8g"),
        arrow_enabled=bool(cfg.get("arrow_enabled", True)),
    )

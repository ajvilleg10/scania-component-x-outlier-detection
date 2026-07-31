from __future__ import annotations

from pyspark.sql import SparkSession


class SparkSessionFactory:
    """Factory responsible for creating Spark sessions in Colab/local environments."""

    @staticmethod
    def create(
        app_name: str = "ScaniaComponentXAnomalyDetection",
        master: str = "local[*]",
        shuffle_partitions: int = 64,
        driver_memory: str = "8g",
        arrow_enabled: bool = True,
    ) -> SparkSession:
        return (
            SparkSession.builder
            .appName(app_name)
            .master(master)
            .config("spark.sql.shuffle.partitions", str(shuffle_partitions))
            .config("spark.driver.memory", driver_memory)
            .config("spark.sql.execution.arrow.pyspark.enabled", str(arrow_enabled).lower())
            .getOrCreate()
        )

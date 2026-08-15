from __future__ import annotations

import csv
from pathlib import Path
from typing import Iterable, List

from pyspark.sql import DataFrame
from pyspark.sql import functions as F


class DataQualityAnalyzer:
    """Performs scalable quality checks over Spark DataFrames.

    The methods return native Python rows (list[dict]) instead of Pandas DataFrames
    to keep the project independent of Pandas and avoid Spark-to-Pandas transfers.
    """

    def __init__(self, dataframe: DataFrame):
        self.df = dataframe

    def shape(self) -> tuple[int, int]:
        return self.df.count(), len(self.df.columns)

    def schema_rows(self) -> list[dict]:
        return [{"column": col_name, "dtype": dtype} for col_name, dtype in self.df.dtypes]

    # Backward-compatible alias used by older code paths. It does not return Pandas.
    def schema_as_rows(self) -> list[dict]:
        return self.schema_rows()

    def duplicated_count(self, subset: Iterable[str] | None = None) -> int:
        if subset:
            return self.df.count() - self.df.dropDuplicates(list(subset)).count()
        return self.df.count() - self.df.dropDuplicates().count()

    def missing_report(self) -> list[dict]:
        total = self.df.count()
        expressions = []
        for col_name, dtype in self.df.dtypes:
            if dtype in {"double", "float"}:
                expr = F.sum(F.when(F.col(col_name).isNull() | F.isnan(F.col(col_name)), 1).otherwise(0)).alias(col_name)
            else:
                expr = F.sum(F.when(F.col(col_name).isNull(), 1).otherwise(0)).alias(col_name)
            expressions.append(expr)

        row = self.df.select(expressions).collect()[0].asDict()
        rows = [
            {
                "column": col,
                "missing_count": int(count or 0),
                "missing_ratio": float((count or 0) / max(total, 1)),
            }
            for col, count in row.items()
        ]
        return sorted(rows, key=lambda r: r["missing_ratio"], reverse=True)

    def constant_columns(self, columns: List[str] | None = None) -> List[str]:
        cols = columns or self.df.columns
        constant = []
        for col_name in cols:
            try:
                n_unique = self.df.select(col_name).distinct().limit(2).count()
                if n_unique <= 1:
                    constant.append(col_name)
            except Exception:
                continue
        return constant


def save_rows_csv(rows: list[dict], path: str | Path, fieldnames: list[str] | None = None) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if not fieldnames:
        keys = []
        for row in rows:
            for key in row.keys():
                if key not in keys:
                    keys.append(key)
        fieldnames = keys
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def save_quality_report(report: list[dict], path: str) -> None:
    """Persist a quality report as CSV."""
    save_rows_csv(report, path, fieldnames=["column", "missing_count", "missing_ratio"])

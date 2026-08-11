from __future__ import annotations

from typing import Iterable, List

import pandas as pd
from pyspark.sql import DataFrame
from pyspark.sql import functions as F


class DataQualityAnalyzer:
    """Performs scalable quality checks over Spark DataFrames."""

    def __init__(self, dataframe: DataFrame):
        self.df = dataframe

    def shape(self) -> tuple[int, int]:
        return self.df.count(), len(self.df.columns)

    def schema_as_pandas(self) -> pd.DataFrame:
        return pd.DataFrame(self.df.dtypes, columns=["column", "dtype"])

    def duplicated_count(self, subset: Iterable[str] | None = None) -> int:
        if subset:
            return self.df.count() - self.df.dropDuplicates(list(subset)).count()
        return self.df.count() - self.df.dropDuplicates().count()

    def missing_report(self) -> pd.DataFrame:
        total = self.df.count()
        expressions = []
        for col_name, dtype in self.df.dtypes:
            if dtype in {"double", "float"}:
                expr = F.sum(F.when(F.col(col_name).isNull() | F.isnan(F.col(col_name)), 1).otherwise(0)).alias(col_name)
            else:
                expr = F.sum(F.when(F.col(col_name).isNull(), 1).otherwise(0)).alias(col_name)
            expressions.append(expr)

        row = self.df.select(expressions).collect()[0].asDict()
        return (
            pd.DataFrame({"column": list(row.keys()), "missing_count": list(row.values())})
            .assign(missing_ratio=lambda x: x["missing_count"] / max(total, 1))
            .sort_values("missing_ratio", ascending=False)
            .reset_index(drop=True)
        )

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


def save_quality_report(report: pd.DataFrame, path: str) -> None:
    """Persist a quality report as CSV."""
    from pathlib import Path

    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    report.to_csv(output_path, index=False)

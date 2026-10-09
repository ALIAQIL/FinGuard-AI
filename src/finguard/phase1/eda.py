"""Run a bounded PySpark profile for the ULB credit-card fraud CSV.

The job writes JSON/CSV summaries and avoids collecting transaction-level data
to the driver. Plots are intentionally a later notebook concern; the first
milestone establishes reproducible, machine-readable profiling outputs.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def _spark_session(master: str):
    try:
        from pyspark.sql import SparkSession
    except ImportError as exc:  # pragma: no cover - environment dependent
        raise RuntimeError("PySpark is required; install requirements.txt first") from exc
    return (
        SparkSession.builder
        .appName("FinGuardAI-Phase1-EDA")
        .master(master)
        .config("spark.sql.shuffle.partitions", "4")
        .config("spark.driver.memory", "1g")
        .getOrCreate()
    )


def profile(input_path: str | Path, output_dir: str | Path, master: str = "local[2]") -> Path:
    """Profile a CSV with Spark and return the output directory."""
    from pyspark.sql import functions as F

    spark = _spark_session(master)
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    try:
        raw = spark.read.option("header", True).option("inferSchema", True).csv(str(input_path))
        row_count = raw.count()
        columns = []
        for field in raw.schema.fields:
            nulls = raw.where(F.col(field.name).isNull()).count()
            distinct = raw.select(field.name).where(F.col(field.name).isNotNull()).distinct().count()
            columns.append({
                "name": field.name,
                "spark_type": field.dataType.simpleString(),
                "null_count": nulls,
                "null_rate": nulls / row_count if row_count else 0.0,
                "distinct_count": distinct,
            })

        numeric = [f.name for f in raw.schema.fields if f.dataType.simpleString() in {"int", "bigint", "double", "float", "long"}]
        numeric_summary = []
        if numeric:
            summary = raw.select(numeric).summary("count", "mean", "stddev", "min", "25%", "50%", "75%", "max").toPandas()
            numeric_summary = summary.to_dict(orient="records")

        fraud_column = next(
            (name for name in raw.columns if name.lower() in {"class", "isfraud"}),
            None,
        )
        fraud_counts = []
        if fraud_column:
            fraud_counts = [row.asDict() for row in raw.groupBy(fraud_column).count().orderBy(fraud_column).collect()]
        duplicate_rows = row_count - raw.dropDuplicates().count()
        report = {
            "input": str(input_path),
            "row_count": row_count,
            "column_count": len(raw.columns),
            "columns": columns,
            "numeric_summary": numeric_summary,
            "fraud_counts": fraud_counts,
            "exact_duplicate_count": duplicate_rows,
            "notes": [
                "Correlation is not causation; inspect fraud-conditioned distributions separately.",
                "Chronological splitting is required before model evaluation when event time is meaningful.",
                "The profiling job is descriptive and does not clean or delete raw data.",
            ],
        }
        (output / "profile.json").write_text(json.dumps(report, indent=2, default=str) + "\n", encoding="utf-8")
        if numeric:
            summary.to_csv(output / "numeric_summary.csv", index=False)
        return output
    finally:
        spark.stop()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--master", default="local[2]")
    args = parser.parse_args()
    print(profile(args.input, args.output, args.master))


if __name__ == "__main__":
    main()

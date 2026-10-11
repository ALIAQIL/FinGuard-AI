"""Consume ULB events from Kafka with Spark Structured Streaming.

Run with Spark's Kafka connector, for example:
spark-submit --packages org.apache.spark:spark-sql-kafka-0-10_2.12:3.5.9 \
  scripts/stream_ulb.py --checkpoint data/checkpoints/ulb --output data/gold/stream_predictions
"""

from __future__ import annotations

import argparse


def main() -> None:
    from pyspark.sql import SparkSession
    from pyspark.sql import functions as F
    from pyspark.sql import types as T

    parser = argparse.ArgumentParser()
    parser.add_argument("--bootstrap-servers", default="localhost:9092")
    parser.add_argument("--topic", default="transactions")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--trigger-seconds", type=int, default=10)
    args = parser.parse_args()
    spark = (
        SparkSession.builder.appName("FinGuardAI-ULB-Streaming")
        .master("local[2]")
        .config("spark.sql.shuffle.partitions", "4")
        .getOrCreate()
    )
    features_schema = T.MapType(T.StringType(), T.DoubleType(), valueContainsNull=False)
    event_schema = T.StructType([
        T.StructField("event_id", T.StringType(), False),
        T.StructField("event_timestamp", T.StringType(), False),
        T.StructField("schema_version", T.StringType(), False),
        T.StructField("features", features_schema, False),
    ])
    events = (
        spark.readStream.format("kafka")
        .option("kafka.bootstrap.servers", args.bootstrap_servers)
        .option("subscribe", args.topic)
        .option("startingOffsets", "latest")
        .option("failOnDataLoss", "false")
        .load()
    )
    parsed = events.select(F.from_json(F.col("value").cast("string"), event_schema).alias("event"))
    predictions = (
        parsed.select("event.*")
        .withColumn("event_time", F.to_timestamp("event_timestamp"))
        .withWatermark("event_time", "10 minutes")
        .dropDuplicates(["event_id"])
        .withColumn("model_version", F.lit("not-scored-yet"))
        .withColumn("risk_score", F.lit(None).cast("double"))
        .withColumn("processed_at", F.current_timestamp())
    )
    query = (
        predictions.writeStream.format("parquet")
        .option("path", args.output)
        .option("checkpointLocation", args.checkpoint)
        .outputMode("append")
        .trigger(processingTime=f"{args.trigger_seconds} seconds")
        .start()
    )
    query.awaitTermination()


if __name__ == "__main__":
    main()

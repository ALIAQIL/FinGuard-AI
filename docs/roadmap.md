# FinGuard AI roadmap

## Phase 1 — EDA (current)

**Theory:** distribution shape, missingness, duplicates, class imbalance, correlation limits, and chronological analysis.

**Design:** preserve raw input; profile with Spark; write machine-readable reports; never delete records during EDA.

**Implementation:** deterministic fixture generator and bounded PySpark profiler.

**Validation:** unit tests for formulas and fixture schema; run the Spark job on the fixture when PySpark is installed.

**Optimization:** use distributed aggregations, four shuffle partitions, `local[2]`, and no transaction-level `collect()`.

**Interview questions:** Why is PR-AUC more informative than accuracy for rare fraud? Why must train/test splits respect event time? Which missingness patterns might be predictive?

## Phase 2 — Data quality and medallion layers

Define a schema contract, completeness/validity/uniqueness/consistency/timeliness checks, quarantine output, and idempotent Bronze → Silver → Gold jobs. Add exact and business-key duplicate tests.

## Phase 3 — Leakage-safe features

Implement point-in-time features using window functions and prior rows only. Add temporal, amount, velocity, and entity-history features, with a feature contract shared by batch and streaming.

## Phase 4 — Modeling

Create chronological train/validation/test splits, logistic baseline, CPU XGBoost histogram model, class-weight experiments, PR-AUC/recall/precision-at-K metrics, and threshold selection using explicit investigation costs.

## Phase 5 — Kafka and Structured Streaming

Add validated JSON events, a replay producer, one local KRaft broker, Spark micro-batches, watermarks, checkpointing, dead-letter handling, and idempotent prediction writes.

## Phase 6 — Airflow, testing, and observability

Orchestrate offline jobs only; add integration tests, restart drills, throughput/latency measurements, and resource profiling. Keep continuously running Kafka/Spark services outside Airflow DAG execution.

## Phase 7 — Interview-ready demo

Package a short, reproducible demo and document measured limitations. No benchmark, model score, or production claim is included without a recorded run and configuration.


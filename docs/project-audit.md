# FinGuard AI project audit

This audit separates verified repository behavior from production capabilities that are not yet implemented. It is intentionally specific so interviewers can see the engineering trade-offs instead of reading unsupported claims.

## Findings

| Severity | Finding | Evidence | Status / remediation |
| --- | --- | --- | --- |
| High | The streaming path does not score transactions. | `scripts/stream_ulb.py` writes `risk_score` as null and `model_version` as `not-scored-yet`. | Documented limitation. Model artifact loading, feature parity, scoring, and an alert sink remain follow-up work. |
| High | Live Kafka delivery and restart behavior are not covered by CI. | CI runs unit tests, the real-data benchmark, and Spark EDA, but no broker integration test. | Documented limitation. Add broker-backed tests before claiming streaming reliability. |
| High | Customer/card behavior cannot be modeled from this dataset. | The public ULB schema contains anonymized PCA fields, `Time`, `Amount`, and `Class`, but no entity identifier. | The feature pipeline does not fabricate customer history. A production project needs an entity-aware source. |
| High | The dataset is not representative of every banking deployment. | It covers two days in September 2013, is anonymized, and contains no issuer, merchant, geography, or current fraud context. | Use it for reproducible engineering evaluation, not current fraud-performance claims. |
| Medium | The `Time` field is elapsed seconds, not a business timestamp. | The feature code derives hour/day from elapsed seconds and converts it to UTC for replay events. | Documented limitation. A production contract needs timezone, event-time semantics, and late-arrival policy. |
| Medium | Duplicate removal affects the fraud population. | The raw dataset contains exact duplicates; the benchmark removes them before chronological splitting. | The report now separates contract-invalid rows from accepted-after-dedup rows and records the policy. A future sensitivity run should compare both policies. |
| Medium | Model selection and operations have different objectives. | F1-only thresholding can produce an alert volume that investigators cannot handle. | The benchmark now selects models by temporal CV PR-AUC and reports a fixed top-1% investigation-capacity policy. |
| Medium | Results can vary across dependency and runtime versions. | Local and CI runs previously produced different thresholds and metrics. | Runtime package metadata is recorded in the benchmark JSON; results remain reproducible experiments, not guarantees. |
| Medium | The Airflow DAG is not a full medallion pipeline. | `dags/finguard_batch.py` currently validates input, profiles data, and runs the benchmark. | Documented as an orchestration skeleton. Bronze/Silver/Gold production jobs are not claimed. |
| Medium | No calibration, drift monitoring, model registry, or serving contract exists. | The repository evaluates offline probabilities and writes reports, but does not publish or monitor a model artifact. | Explicit follow-up work before deployment. |

## What the project can credibly demonstrate

- Real public fraud data ingestion rather than synthetic examples.
- Schema, missing-value, duplicate, target, and amount validation with quarantine output.
- Chronological train/validation/test evaluation.
- Imbalance-aware baseline and CPU XGBoost comparison.
- Time-aware `RandomizedSearchCV` and `GridSearchCV` constrained to the training period.
- PR-AUC model selection plus precision and fraud capture under a declared alert budget.
- Deterministic event IDs, Kafka replay foundations, Spark watermarking, and event deduplication.
- CI evidence for tests, linting, real-data benchmarking, and Spark EDA.

## Interview-safe framing

Present this as a resource-conscious fraud-detection engineering prototype with an honest path toward productionization. Emphasize leakage prevention, metric/business-objective separation, duplicate policy, data limitations, and the boundary between implemented streaming foundations and unimplemented model serving. Do not present the offline test metrics as fraud reduction, live latency, or production availability.

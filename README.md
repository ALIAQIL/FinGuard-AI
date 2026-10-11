# FinGuard AI

FinGuard AI is a fraud-detection engineering project built around the public anonymized ULB/Worldline credit-card dataset. It demonstrates data profiling, data-quality validation, leakage-aware feature preparation, imbalanced classification, Kafka event replay, and Spark Structured Streaming concepts.

## Current status

**Phase 1 is implemented and measured.** The benchmark and CI paths are working. The Kafka and Spark streaming components are functional foundations, but the repository does not claim an end-to-end production scoring service yet.

This distinction is intentional: the README reports what is implemented and verified, while unfinished production components are listed as next steps.

## What is implemented

- Real ULB/Worldline dataset ingestion instructions and schema contract
- PySpark EDA over the full dataset
- Missing-value, schema, duplicate, target, and amount validation
- Quarantine output for invalid rows
- Row-local temporal and amount features
- Chronological 60/20/20 train/validation/test split
- Logistic Regression baseline
- CPU XGBoost benchmark with deterministic time-aware randomized and grid search
- PR-AUC model selection plus a top-1% investigation-capacity operating policy
- Precision, recall, F1, ROC-AUC, PR-AUC, precision-at-K, alert-budget capture, confusion matrix, and timing metrics
- Runtime, dependency, search-space, and cross-validation metadata in the benchmark report
- Deterministic Kafka event schema and CSV replay producer
- Spark Structured Streaming Kafka parser with watermarking and event deduplication
- Airflow DAG skeleton for input validation, EDA, and benchmark execution
- GitHub Actions workflow for tests, real-data benchmark, and Spark EDA

## What is not implemented yet

- The Gold feature set does not contain customer-history features because this dataset has no customer/card identifier.
- The streaming job currently parses, watermarks, deduplicates, and writes events; its `risk_score` is intentionally not populated until model-artifact loading and feature parity are completed.
- Bronze/Silver/Gold Parquet jobs are documented but not yet exposed as separate production-grade commands.
- Kafka integration and streaming restart behavior have not been run against a live broker in CI.
- The Airflow DAG is a local orchestration skeleton, not a complete production scheduler deployment.
- No production claims are made about fraud reduction, availability, latency, compliance, or current fraud behavior.

The evidence-based limitations and suspicious gaps are tracked in [docs/project-audit.md](docs/project-audit.md).

## Architecture

The solid path is implemented today. The dashed path is the intended next integration step.

```mermaid
flowchart LR
    A[ULB / Worldline CSV] --> B[PySpark EDA]
    A --> C[Quality validation]
    C --> D[Row-local features]
    D --> E[Chronological split]
    E --> F[Logistic baseline]
    E --> G[XGBoost benchmark]
    A --> H[Validated event schema]
    H --> I[Kafka replay producer]
    I --> J[Kafka topic]
    J --> K[Spark streaming parser]
    K --> L[Watermark + deduplication]
    L --> M[Parquet streaming output]
    G -. model artifact + shared feature contract .-> K
    K -. risk score + alert sink .-> N[Future scoring service]
```

## Dataset

The project uses the [ULB Machine Learning Group credit-card dataset on Kaggle](https://www.kaggle.com/mlg-ulb/creditcardfraud), also indexed as [OpenML dataset 1597](https://www.openml.org/search?id=1597). It contains 284,807 transactions from two days in September 2013 and 492 fraud labels. `V1`–`V28` are PCA-transformed for confidentiality; `Time`, `Amount`, and `Class` are the remaining interpretable fields.

The dataset has no stable customer/card identifier. Therefore, this project does not invent customer velocity, customer history, merchant history, or card-level behavior features. That is a limitation of the data, not an undocumented assumption.

Download the CSV under the dataset’s published terms and place it at:

```text
data/raw/creditcard.csv
```

The raw file is ignored by Git. The helper script is [scripts/download_ulb_dataset.sh](scripts/download_ulb_dataset.sh).

## Quick start

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e . -r requirements-dev.txt

# After downloading data/raw/creditcard.csv:
PYTHONPATH=src python -m finguard.phase1.eda \
  --input data/raw/creditcard.csv \
  --output reports/phase1 \
  --master 'local[2]'

PYTHONPATH=src python -m finguard.benchmark \
  --input data/raw/creditcard.csv \
  --output reports/benchmark_ulb.json \
  --tune-xgboost \
  --cv-splits 3 \
  --random-search-iterations 4 \
  --alert-rate 0.01

PYTHONPATH=src pytest -q
```

Spark 3.5 is run with Java 17 in CI. The full local validation command is:

```bash
JAVA_HOME=/path/to/java-17 ./scripts/ci_local.sh
```

## Measured benchmark

Configuration: full ULB dataset, exact duplicates removed, chronological 60/20/20 split, class-weighted logistic regression, CPU XGBoost with histogram training, three-fold temporal cross-validation, four randomized trials, four grid combinations, and a top-1% validation alert budget.

| Model | PR-AUC | Precision@1% alerts | Fraud capture@1% | F1 at 1% alerts |
| --- | ---: | ---: | ---: | ---: |
| Logistic Regression | 0.7611 | 0.1127 | 0.8649 | 0.1850 |
| XGBoost baseline | 0.7820 | 0.1127 | 0.8649 | 0.2148 |
| XGBoost randomized search | **0.7888** | **0.1127** | **0.8649** | 0.1928 |
| XGBoost grid search | 0.7649 | 0.1109 | 0.8514 | **0.2062** |

The recorded run selected the randomized-search model by training-only temporal CV PR-AUC (0.7900 versus 0.7766 for the grid search). The test set was evaluated once after the operating policy was selected. These are offline experiments, not universal performance claims; the fixed 1% alert budget is an explicit operating assumption, not a measured business outcome.

Details and measured timings are in [docs/benchmark.md](docs/benchmark.md); the machine-readable output is [reports/benchmark_ulb.json](reports/benchmark_ulb.json).

## Validation and CI

The repository includes [.github/workflows/ci.yml](.github/workflows/ci.yml), which runs:

1. Python compilation, pytest, and Ruff.
2. Real ULB dataset download from the OpenML public mirror when the CI cache is empty.
3. The full benchmark, including both XGBoost searches.
4. Full Spark EDA with Java 17.
5. Benchmark and EDA report artifact upload.

The local equivalent uses the shared `/home/aliaqil/Desktop/projects/.venv` environment. CI artifacts contain the measured benchmark and tuning metadata; no synthetic dataset is used.

## Interview focus

- How duplicate handling changes the fraud population and why it must be documented.
- Why chronological evaluation is safer than a random split for transaction data.
- Why PR-AUC, recall, precision, and investigation capacity matter more than accuracy.
- Why a threshold is selected separately from the model and fitted using validation data.
- Why PR-AUC is used for imbalanced model selection while alert-budget capture and precision express the business objective.
- How RandomizedSearchCV and GridSearchCV are constrained to chronological training folds.
- Why the absence of an entity identifier prevents legitimate customer-history features.
- How Kafka offsets, event IDs, checkpoints, watermarks, late events, and idempotent sinks work together.
- Which parts of this repository are demonstrated locally and which require production infrastructure.

## Repository structure

```text
FinGuardAI/
├── .github/workflows/ci.yml
├── dags/finguard_batch.py
├── data/{raw,bronze,silver,gold,quarantine}/
├── docs/{benchmark,phase-1-eda,roadmap,system-design}.md
├── reports/benchmark_ulb.json
├── scripts/{ci_local.sh,download_ulb_dataset.sh,stream_ulb.py}
├── src/finguard/
│   ├── benchmark.py
│   ├── config.py
│   ├── features.py
│   ├── quality.py
│   ├── phase1/{eda,stats}.py
│   └── streaming/{events,producer}.py
└── tests/
```

## Documentation

- [Benchmark methodology and results](docs/benchmark.md)
- [Phase 1 EDA theory and validation](docs/phase-1-eda.md)
- [System design and trade-offs](docs/system-design.md)
- [Implementation roadmap](docs/roadmap.md)
- [Evidence-based project audit](docs/project-audit.md)
- [Docker Compose Kafka configuration](docker-compose.yml)

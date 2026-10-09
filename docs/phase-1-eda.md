# Phase 1 — Dataset selection and exploratory data analysis

## 1. Recommendation

Use the ULB/Worldline Credit Card Fraud Detection dataset. It contains 284,807 real anonymized credit-card transactions from European cardholders over two days in September 2013, including 492 frauds (0.172%). `V1`–`V28` are PCA-transformed features released to protect confidentiality; `Time` is elapsed seconds, `Amount` is the transaction amount, and `Class` is the fraud label. The source is the [ULB Machine Learning Group Kaggle dataset](https://www.kaggle.com/mlg-ulb/creditcardfraud); it is also indexed as [OpenML 1597](https://www.openml.org/search?id=1597).

Download it using the Kaggle website/CLI under its published terms and store it locally as `data/raw/creditcard.csv`. Do not commit the 150 MB raw file or credentials. IEEE-CIS is a later expansion because its larger, richer feature surface increases memory and interpretation cost. PaySim is intentionally not used because it is synthetic.

## 2. Statistical foundations

For observations (x_1, ldots, x_n):

- **Mean:** \(\bar{x}=n^{-1}\sum_i x_i\). It is the arithmetic center and is useful for average transaction value, but is sensitive to extreme fraud amounts.
- **Median:** the 50th percentile. It is robust to skew and is often a better typical-transaction summary.
- **Sample variance:** \(s^2=\sum_i(x_i-\bar{x})^2/(n-1)\); standard deviation is \(s=\sqrt{s^2}\). These measure dispersion, but do not imply normality.
- **Quantiles/IQR:** \(IQR=Q_3-Q_1\). IQR is robust for flagging unusually large amounts, but fraud can be a genuine tail event and must not be deleted automatically.
- **Skewness:** a standardized third central moment; positive skew indicates a long upper tail, common for monetary amounts. **Kurtosis** is based on the fourth standardized central moment and describes tail heaviness; conventions differ on excess versus non-excess kurtosis.
- **Correlation:** Pearson \(r=\operatorname{cov}(X,Y)/(\sigma_X\sigma_Y)\) measures linear association. Spearman correlation uses ranks and can reveal monotonic relationships. Neither proves causation and both can be unstable when fraud is rare.
- **Class imbalance:** if \(p\) is the fraud prevalence, \(p=N_{fraud}/N\). Accuracy can look excellent for a useless all-legitimate classifier when \(p\) is small, so use precision, recall, and PR-AUC later.

Spark equivalents include `avg`, `stddev_samp`, `percentile_approx`, `skewness`, `kurtosis`, `corr`, `count`, and `countDistinct`. They run as distributed aggregations; exact percentiles may be expensive, so `percentile_approx` is appropriate for a laptop profile.

## 3. What the first profile measures

The initial job reports row/column counts, inferred data types, null counts and rates, distinct counts, numeric descriptive statistics, fraud-label counts, and exact duplicates. It is deliberately descriptive: it does not mutate raw data or pretend that an inferred schema is a validated production contract.

Run:

```bash
python -m finguard.phase1.eda --input data/raw/creditcard.csv --output reports/phase1 --master local[2]
```

Expected outputs:

- `reports/phase1/profile.json`: machine-readable profile and data-quality observations.
- `reports/phase1/numeric_summary.csv`: Spark summary rows for numeric columns.

The exact values depend on the downloaded file and the profiling configuration. The repository intentionally does not publish invented benchmark results.

## 4. Missingness, duplicates, and outliers

Missing completely at random (MCAR) is unrelated to observed and unobserved values; missing at random (MAR) depends on observed values; missing not at random (MNAR) depends on the missing value or an unobserved process. Missingness indicators can be predictive—for example, a device field omitted only on a risky channel—so imputation and indicator policy must be learned on training data and documented.

Exact duplicate rows are different from duplicate business keys. A repeated legitimate payment can share an amount and customer but have a different event ID. Kafka redelivery is an infrastructure duplicate and needs an event ID plus an idempotent sink. Outlier methods (z-score, IQR, percentile caps) are diagnostic tools. Domain checks such as negative amount or impossible balance transitions are stronger validity rules. Fraudulent transactions should be investigated, not silently removed as outliers.

## 5. Phase 1 design and leakage boundary

EDA may compare fraud and non-fraud distributions because labels are available historically. It must not turn a full-dataset statistic into an online feature. Any mean, standard deviation, frequency map, encoder, threshold, or imputation value used by a model must be fitted on training data only. Behavioral features for transaction (t) must use transactions strictly before (t), not a centered window or future label.

The ULB dataset does not expose a customer/card identifier, so customer-history, merchant-history, and per-card velocity features cannot be constructed legitimately from it. Do not fabricate an entity key. The project will focus on temporal, amount, PCA-space, and label-aware evaluation features, and will clearly mark entity-level streaming features as unavailable for this dataset. A future approved dataset with stable entity IDs can extend that part of the design.

## 6. Phase 1 validation and interview prompts

The tests verify statistical definitions and the expected real-dataset schema contract. A Spark integration test will be added once the project’s dependency matrix is pinned and CI has a Java/PySpark runtime.

Interview prompts:

1. Why can accuracy be misleading for fraud? Because the majority class dominates; a classifier can be accurate while missing nearly every fraud case.
2. Why use chronological splitting? Random splitting can let future behavior influence training and produce optimistic estimates under drift.
3. Why not delete IQR outliers? Fraud often lives in the tail; deleting it changes the target distribution and may remove the very signal being detected.
4. Why avoid `collect()`? It moves data to the driver and can exceed the small laptop’s memory; aggregate or write distributed outputs instead.

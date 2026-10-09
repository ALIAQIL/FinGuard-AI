"""Train and measure a baseline plus CPU XGBoost on real ULB data."""

from __future__ import annotations

import argparse
import json
import time
import urllib.request
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from finguard.config import ULB_FEATURE_COLUMNS, ULB_TARGET
from finguard.features import add_ulb_features
from finguard.quality import clean_ulb_frame


def _metrics(y_true: pd.Series, probabilities: np.ndarray, threshold: float) -> dict[str, Any]:
    predictions = (probabilities >= threshold).astype("int8")
    tn, fp, fn, tp = confusion_matrix(y_true, predictions, labels=[0, 1]).ravel()
    k = max(1, int(y_true.sum()))
    top_indices = np.argsort(-probabilities)[:k]
    precision_at_k = float(np.asarray(y_true)[top_indices].sum() / k)
    return {
        "threshold": threshold,
        "precision": float(precision_score(y_true, predictions, zero_division=0)),
        "recall": float(recall_score(y_true, predictions, zero_division=0)),
        "f1": float(f1_score(y_true, predictions, zero_division=0)),
        "roc_auc": float(roc_auc_score(y_true, probabilities)),
        "pr_auc": float(average_precision_score(y_true, probabilities)),
        "precision_at_k": precision_at_k,
        "confusion_matrix": {"tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp)},
    }


def _best_threshold(y_true: pd.Series, probabilities: np.ndarray) -> float:
    thresholds = np.linspace(0.01, 0.99, 99)
    scores = [f1_score(y_true, probabilities >= threshold, zero_division=0) for threshold in thresholds]
    return float(thresholds[int(np.argmax(scores))])


def _load_openml(path: Path) -> Path:
    parquet_path = path.with_suffix(".parquet")
    urllib.request.urlretrieve(
        "https://data.openml.org/datasets/0000/1597/dataset_1597.pq",
        parquet_path,
    )
    frame = pd.read_parquet(parquet_path)
    frame.columns = [str(column) for column in frame.columns]
    frame = frame.rename(columns={column: "Class" for column in frame.columns if column.lower() == "class"})
    frame.to_csv(path, index=False)
    parquet_path.unlink(missing_ok=True)
    return path


def run_benchmark(input_path: str | Path, output_path: str | Path, max_rows: int | None = None) -> dict[str, Any]:
    started = time.perf_counter()
    path = Path(input_path)
    if not path.exists():
        raise FileNotFoundError(f"Dataset not found: {path}. Run scripts/download_ulb_dataset.sh first.")
    frame = pd.read_csv(path, nrows=max_rows)
    accepted, quarantine, quality = clean_ulb_frame(frame)
    accepted = add_ulb_features(accepted).sort_values("Time").reset_index(drop=True)
    if accepted[ULB_TARGET].nunique() < 2:
        raise ValueError("Benchmark input must contain both classes")

    feature_columns = [*ULB_FEATURE_COLUMNS, "event_hour", "event_day", "amount_log1p"]
    x = accepted[feature_columns].astype("float32")
    y = accepted[ULB_TARGET].astype("int8")
    n = len(accepted)
    train_end, valid_end = int(n * 0.6), int(n * 0.8)
    x_train, x_valid, x_test = x.iloc[:train_end], x.iloc[train_end:valid_end], x.iloc[valid_end:]
    y_train, y_valid, y_test = y.iloc[:train_end], y.iloc[train_end:valid_end], y.iloc[valid_end:]
    positive = max(1, int(y_train.sum()))
    negative = max(1, int((y_train == 0).sum()))

    models: dict[str, Any] = {
        "logistic_regression": Pipeline([
            ("scale", StandardScaler()),
            ("model", LogisticRegression(max_iter=300, class_weight="balanced", solver="lbfgs")),
        ]),
    }
    try:
        from xgboost import XGBClassifier

        models["xgboost"] = XGBClassifier(
            n_estimators=120,
            max_depth=4,
            learning_rate=0.08,
            subsample=0.8,
            colsample_bytree=0.8,
            min_child_weight=2,
            reg_lambda=2.0,
            objective="binary:logistic",
            eval_metric="aucpr",
            tree_method="hist",
            n_jobs=2,
            random_state=7,
            scale_pos_weight=negative / positive,
        )
    except ImportError:
        models["xgboost_unavailable"] = None

    results: dict[str, Any] = {
        "dataset": str(path),
        "rows_read": len(frame),
        "rows_accepted": len(accepted),
        "rows_quarantined": len(quarantine),
        "fraud_count": int(y.sum()),
        "fraud_rate": float(y.mean()),
        "quality": quality.as_dict(),
        "split": {"train": len(x_train), "validation": len(x_valid), "test": len(x_test), "strategy": "chronological 60/20/20"},
        "models": {},
    }
    for name, model in models.items():
        if model is None:
            results["models"][name] = {"status": "not installed"}
            continue
        fit_started = time.perf_counter()
        model.fit(x_train, y_train)
        train_seconds = time.perf_counter() - fit_started
        valid_probabilities = model.predict_proba(x_valid)[:, 1]
        threshold = _best_threshold(y_valid, valid_probabilities)
        score_started = time.perf_counter()
        test_probabilities = model.predict_proba(x_test)[:, 1]
        score_seconds = time.perf_counter() - score_started
        test_metrics = _metrics(y_test, test_probabilities, threshold)
        test_metrics["inference_ms_per_row"] = score_seconds * 1000 / max(1, len(x_test))
        results["models"][name] = {
            "status": "ok",
            "train_seconds": train_seconds,
            "test_score_seconds": score_seconds,
            "validation_threshold": threshold,
            "test": test_metrics,
        }
    results["elapsed_seconds"] = time.perf_counter() - started
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(results, indent=2) + "\n", encoding="utf-8")
    return results


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", default="data/raw/creditcard.csv")
    parser.add_argument("--output", default="reports/benchmark_ulb.json")
    parser.add_argument("--max-rows", type=int)
    args = parser.parse_args()
    result = run_benchmark(args.input, args.output, args.max_rows)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()

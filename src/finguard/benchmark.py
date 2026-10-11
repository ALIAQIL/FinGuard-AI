"""Train and measure leakage-aware fraud models on real ULB data."""

from __future__ import annotations

import argparse
import importlib.metadata
import json
import math
import platform
import sys
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
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV, TimeSeriesSplit
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from finguard.config import ULB_FEATURE_COLUMNS, ULB_TARGET
from finguard.features import add_ulb_features
from finguard.quality import clean_ulb_frame

RANDOM_SEARCH_DISTRIBUTIONS: dict[str, list[Any]] = {
    "n_estimators": [80, 120, 160, 200],
    "max_depth": [3, 4, 5, 6],
    "learning_rate": [0.03, 0.05, 0.08, 0.12],
    "min_child_weight": [1, 2, 5, 10],
    "subsample": [0.7, 0.8, 0.9, 1.0],
    "colsample_bytree": [0.7, 0.8, 0.9, 1.0],
    "reg_alpha": [0.0, 0.1, 0.5],
    "reg_lambda": [1.0, 2.0, 5.0, 10.0],
}

GRID_SEARCH_GRID: dict[str, list[Any]] = {
    "max_depth": [3, 5],
    "min_child_weight": [1, 5],
    "n_estimators": [120],
    "learning_rate": [0.08],
}


def _classification_metrics(y_true: pd.Series, predictions: np.ndarray) -> dict[str, Any]:
    tn, fp, fn, tp = confusion_matrix(y_true, predictions, labels=[0, 1]).ravel()
    return {
        "precision": float(precision_score(y_true, predictions, zero_division=0)),
        "recall": float(recall_score(y_true, predictions, zero_division=0)),
        "f1": float(f1_score(y_true, predictions, zero_division=0)),
        "confusion_matrix": {"tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp)},
    }


def _top_k_indices(probabilities: np.ndarray, alert_rate: float) -> np.ndarray:
    k = max(1, int(math.ceil(len(probabilities) * alert_rate)))
    return np.argsort(-probabilities, kind="stable")[:k]


def _top_k_threshold(probabilities: np.ndarray, alert_rate: float) -> float:
    return float(probabilities[_top_k_indices(probabilities, alert_rate)[-1]])


def _metrics(
    y_true: pd.Series,
    probabilities: np.ndarray,
    threshold: float,
    alert_rate: float,
) -> dict[str, Any]:
    predictions = (probabilities >= threshold).astype("int8")
    metrics = _classification_metrics(y_true, predictions)
    y_values = np.asarray(y_true)
    k = max(1, int(y_values.sum()))
    top_indices = np.argsort(-probabilities, kind="stable")[:k]
    precision_at_k = float(y_values[top_indices].sum() / k)
    budget_indices = _top_k_indices(probabilities, alert_rate)
    budget_labels = y_values[budget_indices]
    budget_k = len(budget_indices)
    metrics.update(
        {
            "threshold": threshold,
            "roc_auc": float(roc_auc_score(y_true, probabilities)),
            "pr_auc": float(average_precision_score(y_true, probabilities)),
            "precision_at_k": precision_at_k,
            "alert_budget_rate": alert_rate,
            "alert_budget_count": budget_k,
            "alerts_per_1000": float(budget_k / max(1, len(y_values)) * 1000),
            "precision_at_alert_budget": float(budget_labels.sum() / budget_k),
            "fraud_capture_at_alert_budget": float(budget_labels.sum() / max(1, int(y_values.sum()))),
        }
    )
    return metrics


def _best_threshold(y_true: pd.Series, probabilities: np.ndarray) -> float:
    thresholds = np.linspace(0.01, 0.99, 99)
    scores = [f1_score(y_true, probabilities >= threshold, zero_division=0) for threshold in thresholds]
    return float(thresholds[int(np.argmax(scores))])


def _validate_cv_folds(y: pd.Series, cv: TimeSeriesSplit) -> None:
    """Fail clearly when a temporal fold cannot calculate average precision."""
    for fold, (_, validation_indices) in enumerate(cv.split(np.zeros(len(y)), y), start=1):
        if y.iloc[validation_indices].nunique() < 2:
            raise ValueError(
                f"TimeSeriesSplit fold {fold} contains one target class; "
                "reduce cv_splits or provide a longer training period."
            )


def _build_xgboost(negative: int, positive: int, **params: Any) -> Any:
    from xgboost import XGBClassifier

    return XGBClassifier(
        objective="binary:logistic",
        eval_metric="aucpr",
        tree_method="hist",
        n_jobs=2,
        random_state=7,
        verbosity=0,
        scale_pos_weight=negative / positive,
        **params,
    )


def _json_value(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _json_value(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_value(item) for item in value]
    if isinstance(value, np.generic):
        return value.item()
    return value


def _search_summary(search: Any) -> dict[str, Any]:
    results = search.cv_results_
    order = np.argsort(results["rank_test_score"])[:5]
    top_results = []
    for index in order:
        top_results.append(
            {
                "rank": int(results["rank_test_score"][index]),
                "mean_test_average_precision": float(results["mean_test_score"][index]),
                "std_test_average_precision": float(results["std_test_score"][index]),
                "params": _json_value(results["params"][index]),
            }
        )
    return {
        "best_score": float(search.best_score_),
        "best_params": _json_value(search.best_params_),
        "trials": int(len(results["params"])),
        "top_results": top_results,
    }


def _tune_xgboost(
    x_train: pd.DataFrame,
    y_train: pd.Series,
    negative: int,
    positive: int,
    cv_splits: int,
    random_search_iterations: int,
) -> tuple[dict[str, Any], Any, Any, str]:
    if cv_splits < 2:
        raise ValueError("cv_splits must be at least 2")
    if random_search_iterations < 1:
        raise ValueError("random_search_iterations must be at least 1")

    cv = TimeSeriesSplit(n_splits=cv_splits)
    _validate_cv_folds(y_train, cv)
    base = _build_xgboost(negative, positive)
    random_search = RandomizedSearchCV(
        estimator=base,
        param_distributions=RANDOM_SEARCH_DISTRIBUTIONS,
        n_iter=random_search_iterations,
        scoring="average_precision",
        cv=cv,
        refit=True,
        random_state=7,
        n_jobs=2,
        pre_dispatch=2,
        return_train_score=True,
    )
    random_started = time.perf_counter()
    random_search.fit(x_train, y_train)
    random_seconds = time.perf_counter() - random_started

    grid_search = GridSearchCV(
        estimator=base,
        param_grid=GRID_SEARCH_GRID,
        scoring="average_precision",
        cv=cv,
        refit=True,
        n_jobs=2,
        pre_dispatch=2,
        return_train_score=True,
    )
    grid_started = time.perf_counter()
    grid_search.fit(x_train, y_train)
    grid_seconds = time.perf_counter() - grid_started

    candidates = {"randomized_search": random_search, "grid_search": grid_search}
    selected_name = max(
        candidates,
        key=lambda name: (
            candidates[name].best_score_,
            -float(np.std(candidates[name].cv_results_["mean_test_score"])),
            name == "grid_search",
        ),
    )
    metadata = {
        "enabled": True,
        "selection_metric": "average_precision",
        "cv": {"type": "TimeSeriesSplit", "n_splits": cv_splits},
        "randomized_search": {
            **_search_summary(random_search),
            "elapsed_seconds": random_seconds,
            "parameter_space": _json_value(RANDOM_SEARCH_DISTRIBUTIONS),
        },
        "grid_search": {
            **_search_summary(grid_search),
            "elapsed_seconds": grid_seconds,
            "parameter_grid": _json_value(GRID_SEARCH_GRID),
        },
        "selected_search": selected_name,
    }
    return metadata, random_search.best_estimator_, grid_search.best_estimator_, selected_name


def _evaluate_model(
    model: Any,
    x_valid: pd.DataFrame,
    y_valid: pd.Series,
    x_test: pd.DataFrame,
    y_test: pd.Series,
    alert_rate: float,
    train_seconds: float | None = None,
) -> dict[str, Any]:
    valid_probabilities = model.predict_proba(x_valid)[:, 1]
    f1_threshold = _best_threshold(y_valid, valid_probabilities)
    operating_threshold = _top_k_threshold(valid_probabilities, alert_rate)
    score_started = time.perf_counter()
    test_probabilities = model.predict_proba(x_test)[:, 1]
    score_seconds = time.perf_counter() - score_started
    result = {
        "status": "ok",
        "train_seconds": train_seconds,
        "test_score_seconds": score_seconds,
        "validation": {
            "pr_auc": float(average_precision_score(y_valid, valid_probabilities)),
            "f1_threshold": f1_threshold,
            "alert_budget_threshold": operating_threshold,
        },
        "validation_threshold": operating_threshold,
        "validation_f1_threshold": f1_threshold,
        "test": _metrics(y_test, test_probabilities, operating_threshold, alert_rate),
        "test_at_f1_threshold": _classification_metrics(
            y_test, (test_probabilities >= f1_threshold).astype("int8")
        ),
    }
    result["test"]["inference_ms_per_row"] = score_seconds * 1000 / max(1, len(x_test))
    return result


def _package_version(name: str) -> str | None:
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return None


def _runtime_metadata() -> dict[str, Any]:
    return {
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "packages": {
            name: _package_version(name)
            for name in ("numpy", "pandas", "scikit-learn", "xgboost")
        },
    }


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


def run_benchmark(
    input_path: str | Path,
    output_path: str | Path,
    max_rows: int | None = None,
    tune_xgboost: bool = False,
    cv_splits: int = 3,
    random_search_iterations: int = 4,
    alert_rate: float = 0.01,
) -> dict[str, Any]:
    started = time.perf_counter()
    if not 0 < alert_rate <= 1:
        raise ValueError("alert_rate must be greater than 0 and at most 1")
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

    results: dict[str, Any] = {
        "dataset": str(path),
        "rows_read": len(frame),
        "rows_accepted": len(accepted),
        "rows_quarantined": len(quarantine),
        "fraud_count": int(y.sum()),
        "fraud_rate": float(y.mean()),
        "quality": quality.as_dict(),
        "deduplication_policy": "exact duplicates removed before chronological sorting and splitting",
        "business_objective": {
            "name": "maximize fraud capture within a fixed investigation capacity",
            "alert_rate": alert_rate,
            "selection_metric": "average_precision",
        },
        "split": {
            "train": len(x_train),
            "validation": len(x_valid),
            "test": len(x_test),
            "strategy": "chronological 60/20/20",
        },
        "runtime": _runtime_metadata(),
        "tuning": {"enabled": False},
        "models": {},
    }

    logistic = Pipeline(
        [
            ("scale", StandardScaler()),
            ("model", LogisticRegression(max_iter=300, class_weight="balanced", solver="lbfgs")),
        ]
    )
    fit_started = time.perf_counter()
    logistic.fit(x_train, y_train)
    results["models"]["logistic_regression"] = _evaluate_model(
        logistic,
        x_valid,
        y_valid,
        x_test,
        y_test,
        alert_rate,
        train_seconds=time.perf_counter() - fit_started,
    )

    try:
        baseline_xgboost = _build_xgboost(
            negative,
            positive,
            n_estimators=120,
            max_depth=4,
            learning_rate=0.08,
            subsample=0.8,
            colsample_bytree=0.8,
            min_child_weight=2,
            reg_lambda=2.0,
        )
    except ImportError:
        results["models"]["xgboost"] = {"status": "not installed"}
        output = Path(output_path)
        output.parent.mkdir(parents=True, exist_ok=True)
        results["elapsed_seconds"] = time.perf_counter() - started
        output.write_text(json.dumps(results, indent=2) + "\n", encoding="utf-8")
        return results

    fit_started = time.perf_counter()
    baseline_xgboost.fit(x_train, y_train)
    results["models"]["xgboost"] = _evaluate_model(
        baseline_xgboost,
        x_valid,
        y_valid,
        x_test,
        y_test,
        alert_rate,
        train_seconds=time.perf_counter() - fit_started,
    )

    if tune_xgboost:
        tuning, random_model, grid_model, selected_name = _tune_xgboost(
            x_train,
            y_train,
            negative,
            positive,
            cv_splits,
            random_search_iterations,
        )
        results["tuning"] = tuning
        results["models"]["xgboost_randomized_search"] = _evaluate_model(
            random_model,
            x_valid,
            y_valid,
            x_test,
            y_test,
            alert_rate,
            train_seconds=tuning["randomized_search"]["elapsed_seconds"],
        )
        results["models"]["xgboost_grid_search"] = _evaluate_model(
            grid_model,
            x_valid,
            y_valid,
            x_test,
            y_test,
            alert_rate,
            train_seconds=tuning["grid_search"]["elapsed_seconds"],
        )
        selected_model = random_model if selected_name == "randomized_search" else grid_model
        results["models"]["xgboost_selected"] = {
            **_evaluate_model(
                selected_model,
                x_valid,
                y_valid,
                x_test,
                y_test,
                alert_rate,
                train_seconds=tuning[selected_name]["elapsed_seconds"],
            ),
            "selected_from": selected_name,
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
    parser.add_argument("--tune-xgboost", action="store_true")
    parser.add_argument("--cv-splits", type=int, default=3)
    parser.add_argument("--random-search-iterations", type=int, default=4)
    parser.add_argument("--alert-rate", type=float, default=0.01)
    args = parser.parse_args()
    result = run_benchmark(
        args.input,
        args.output,
        args.max_rows,
        args.tune_xgboost,
        args.cv_splits,
        args.random_search_iterations,
        args.alert_rate,
    )
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()

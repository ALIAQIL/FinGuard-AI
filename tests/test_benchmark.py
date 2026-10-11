import numpy as np
import pandas as pd
import pytest

from finguard import benchmark
from finguard.config import ULB_COLUMNS


def test_search_spaces_are_deterministic_and_grid_has_four_trials():
    assert benchmark.RANDOM_SEARCH_DISTRIBUTIONS["max_depth"] == [3, 4, 5, 6]
    assert benchmark.GRID_SEARCH_GRID["max_depth"] == [3, 5]
    assert benchmark.GRID_SEARCH_GRID["min_child_weight"] == [1, 5]
    assert np.prod([len(values) for values in benchmark.GRID_SEARCH_GRID.values()]) == 4


def test_time_series_validation_is_chronological():
    y = pd.Series([0, 1] * 20, dtype="int8")
    cv = benchmark.TimeSeriesSplit(n_splits=3)
    for train_indices, validation_indices in cv.split(np.zeros(len(y)), y):
        assert train_indices.max() < validation_indices.min()


def test_time_series_validation_rejects_single_class_fold():
    y = pd.Series([0] * 18 + [1] * 2, dtype="int8")
    with pytest.raises(ValueError, match="one target class"):
        benchmark._validate_cv_folds(y, benchmark.TimeSeriesSplit(n_splits=3))


def test_metrics_report_fixed_alert_budget():
    y = pd.Series([0, 1, 0, 1], dtype="int8")
    probabilities = np.array([0.1, 0.9, 0.8, 0.2])
    result = benchmark._metrics(y, probabilities, threshold=0.8, alert_rate=0.5)
    assert result["alert_budget_count"] == 2
    assert result["alerts_per_1000"] == 500.0
    assert result["precision_at_alert_budget"] == 0.5
    assert result["fraud_capture_at_alert_budget"] == 0.5


class _StubModel:
    def fit(self, x, y):
        return self

    def predict_proba(self, x):
        score = np.asarray(x["Amount"], dtype=float)
        score = (score - score.min()) / max(1.0, score.max() - score.min())
        return np.column_stack([1.0 - score, score])


def _small_frame():
    rows = []
    for index in range(40):
        row = {column: 0.0 for column in ULB_COLUMNS}
        row["Time"] = float(index)
        row["Amount"] = float(index + 1)
        row["Class"] = index % 2
        rows.append(row)
    return pd.DataFrame(rows, columns=ULB_COLUMNS)


def test_benchmark_output_keeps_baseline_and_tuning_metadata(monkeypatch, tmp_path):
    monkeypatch.setattr(benchmark, "_build_xgboost", lambda *args, **kwargs: _StubModel())
    monkeypatch.setattr(
        benchmark,
        "_tune_xgboost",
        lambda *args, **kwargs: (
            {
                "enabled": True,
                "selected_search": "grid_search",
                "selection_metric": "average_precision",
                "randomized_search": {"elapsed_seconds": 0.01},
                "grid_search": {"elapsed_seconds": 0.01},
            },
            _StubModel(),
            _StubModel(),
            "grid_search",
        ),
    )
    input_path = tmp_path / "creditcard.csv"
    output_path = tmp_path / "benchmark.json"
    _small_frame().to_csv(input_path, index=False)

    result = benchmark.run_benchmark(input_path, output_path, tune_xgboost=True)

    assert output_path.exists()
    assert result["tuning"]["enabled"] is True
    assert result["models"]["xgboost"]["status"] == "ok"
    assert result["models"]["xgboost_selected"]["selected_from"] == "grid_search"
    assert result["business_objective"]["alert_rate"] == 0.01

"""Small, dependency-light statistical helpers used by Phase 1.

The Spark job uses Spark's distributed aggregations for the actual dataset. These
helpers keep definitions explicit and make edge cases easy to unit test.
"""

from __future__ import annotations

from math import sqrt
from statistics import median
from typing import Iterable, Sequence


def _values(values: Iterable[float]) -> list[float]:
    result = [float(value) for value in values]
    if not result:
        raise ValueError("at least one numeric value is required")
    return result


def mean(values: Iterable[float]) -> float:
    xs = _values(values)
    return sum(xs) / len(xs)


def sample_variance(values: Iterable[float]) -> float:
    xs = _values(values)
    if len(xs) < 2:
        raise ValueError("sample variance requires at least two values")
    avg = mean(xs)
    return sum((x - avg) ** 2 for x in xs) / (len(xs) - 1)


def sample_stddev(values: Iterable[float]) -> float:
    return sqrt(sample_variance(values))


def iqr(values: Iterable[float]) -> float:
    xs = sorted(_values(values))
    midpoint = len(xs) // 2
    lower = xs[:midpoint]
    upper = xs[(len(xs) + 1) // 2 :]
    return median(upper) - median(lower)


def z_score(value: float, values: Sequence[float]) -> float:
    sd = sample_stddev(values)
    return (float(value) - mean(values)) / sd if sd else 0.0


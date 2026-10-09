import pytest

from finguard.phase1.stats import iqr, mean, sample_stddev, sample_variance, z_score


def test_mean_and_sample_variance_use_standard_definitions():
    assert mean([1, 2, 3, 4]) == pytest.approx(2.5)
    assert sample_variance([1, 2, 3, 4]) == pytest.approx(5 / 3)
    assert sample_stddev([1, 2, 3, 4]) == pytest.approx((5 / 3) ** 0.5)


def test_iqr_is_q3_minus_q1():
    assert iqr([1, 2, 3, 4, 5, 6, 7, 8]) == pytest.approx(4.0)


def test_z_score_is_zero_for_constant_values():
    assert z_score(4, [4, 4, 4]) == 0.0


def test_empty_and_singleton_inputs_are_rejected_where_needed():
    with pytest.raises(ValueError):
        mean([])
    with pytest.raises(ValueError):
        sample_variance([1])


"""Tests for biascompare.stats: t-tests, wilcoxon, bootstrap CI, bonferroni."""

from __future__ import annotations

import math

import numpy as np
import pytest

from biascompare.stats import (
    bonferroni,
    bootstrap_ci,
    mean_or_nan,
    paired_ttest_greater,
    ttest_two_sided,
    wilcoxon_two_sided,
)

# ---------------------------------------------------------------------------
# mean_or_nan
# ---------------------------------------------------------------------------


def test_mean_or_nan_empty_is_nan():
    assert math.isnan(mean_or_nan([]))


def test_mean_or_nan_known_value():
    assert mean_or_nan([1.0, 2.0, 3.0]) == pytest.approx(2.0)


# ---------------------------------------------------------------------------
# paired_ttest_greater
# ---------------------------------------------------------------------------


def test_paired_ttest_greater_nan_for_n_less_than_2():
    assert math.isnan(paired_ttest_greater([]))
    assert math.isnan(paired_ttest_greater([0.5]))


def test_paired_ttest_greater_nan_for_zero_variance():
    assert math.isnan(paired_ttest_greater([0.3, 0.3, 0.3]))


def test_paired_ttest_greater_small_p_for_clearly_positive_mean():
    diffs = [0.10, 0.12, 0.09, 0.11, 0.13, 0.08, 0.10, 0.11]
    p = paired_ttest_greater(diffs)
    assert not math.isnan(p)
    assert 0.0 <= p < 0.01


def test_paired_ttest_greater_large_p_for_negative_mean():
    diffs = [-0.10, -0.12, -0.09, -0.11, -0.13, -0.08]
    p = paired_ttest_greater(diffs)
    assert not math.isnan(p)
    assert p > 0.9


# ---------------------------------------------------------------------------
# ttest_two_sided
# ---------------------------------------------------------------------------


def test_ttest_two_sided_nan_for_n_less_than_2():
    assert math.isnan(ttest_two_sided([]))
    assert math.isnan(ttest_two_sided([1.0]))


def test_ttest_two_sided_nan_for_zero_variance():
    assert math.isnan(ttest_two_sided([0.2, 0.2, 0.2, 0.2]))


def test_ttest_two_sided_small_p_for_mean_far_from_zero():
    values = [1.0, 1.1, 0.9, 1.05, 0.95, 1.02]
    p = ttest_two_sided(values)
    assert not math.isnan(p)
    assert p < 0.01


def test_ttest_two_sided_large_p_when_mean_near_zero():
    values = [0.01, -0.02, 0.015, -0.01, 0.005, -0.005]
    p = ttest_two_sided(values)
    assert not math.isnan(p)
    assert p > 0.5


# ---------------------------------------------------------------------------
# wilcoxon_two_sided
# ---------------------------------------------------------------------------


def test_wilcoxon_two_sided_nan_for_n_less_than_2():
    assert math.isnan(wilcoxon_two_sided([]))
    assert math.isnan(wilcoxon_two_sided([0.4]))


def test_wilcoxon_two_sided_nan_for_all_zero_diffs():
    assert math.isnan(wilcoxon_two_sided([0.0, 0.0, 0.0]))


def test_wilcoxon_two_sided_known_case():
    diffs = [0.5, 0.6, -0.1, 0.4, 0.55, 0.45, -0.2, 0.5]
    p = wilcoxon_two_sided(diffs)
    assert not math.isnan(p)
    assert 0.0 <= p <= 1.0


# ---------------------------------------------------------------------------
# bootstrap_ci
# ---------------------------------------------------------------------------


def test_bootstrap_ci_empty_returns_nan_pair():
    rng = np.random.default_rng(0)
    lower, upper = bootstrap_ci([], rng)
    assert math.isnan(lower)
    assert math.isnan(upper)


def test_bootstrap_ci_single_value_returns_that_value_twice():
    rng = np.random.default_rng(0)
    lower, upper = bootstrap_ci([3.5], rng)
    assert lower == pytest.approx(3.5)
    assert upper == pytest.approx(3.5)


def test_bootstrap_ci_contains_sample_mean_and_is_ordered():
    rng = np.random.default_rng(42)
    values = [1.0, 2.0, 3.0, 4.0, 5.0, 2.5, 3.5]
    lower, upper = bootstrap_ci(values, rng, n_resamples=2000)
    assert lower <= np.mean(values) <= upper
    assert lower <= upper


def test_bootstrap_ci_is_deterministic_given_seeded_rng():
    values = [1.0, 2.0, 3.0, 4.0]
    result_a = bootstrap_ci(values, np.random.default_rng(7), n_resamples=500)
    result_b = bootstrap_ci(values, np.random.default_rng(7), n_resamples=500)
    assert result_a == result_b


# ---------------------------------------------------------------------------
# bonferroni
# ---------------------------------------------------------------------------


def test_bonferroni_multiplies_and_caps_at_one():
    assert bonferroni(0.01, 3) == pytest.approx(0.03)
    assert bonferroni(0.5, 5) == pytest.approx(1.0)


def test_bonferroni_nan_passthrough():
    assert math.isnan(bonferroni(float("nan"), 4))


def test_bonferroni_single_test_is_identity():
    assert bonferroni(0.03, 1) == pytest.approx(0.03)

"""Offline tests for ab/stats.py: Wilson, Newcombe and the seeded percentile bootstrap."""

from __future__ import annotations

import pytest

from ab.stats import (
    bootstrap_means,
    excludes_zero,
    newcombe,
    quantile,
    wilson,
    z_for,
)


def test_z_for():
    assert z_for(0.95) == pytest.approx(1.959964, abs=1e-6)
    assert z_for(0.90) == pytest.approx(1.644854, abs=1e-6)
    with pytest.raises(ValueError):
        z_for(1.0)


def test_wilson_known_values():
    # Newcombe (1998), Table I examples
    assert wilson(81, 263) == pytest.approx((0.2553, 0.3662), abs=1e-4)
    assert wilson(15, 148) == pytest.approx((0.0624, 0.1605), abs=1e-4)
    # boundaries stay inside [0, 1] and are exact at 0/n and n/n
    lo, hi = wilson(0, 10)
    assert lo == 0.0 and hi == pytest.approx(0.2775, abs=1e-4)
    lo, hi = wilson(10, 10)
    assert hi == 1.0 and lo == pytest.approx(0.7225, abs=1e-4)
    assert wilson(0, 0) is None
    with pytest.raises(ValueError):
        wilson(3, 2)


def test_newcombe_known_values():
    # Newcombe (1998) example: 56/70 vs 48/80, difference 0.2 -> (0.0524, 0.3339)
    assert newcombe(48, 80, 56, 70) == pytest.approx((0.0524, 0.3339), abs=1e-4)
    # b - a orientation: swapping arms mirrors the interval
    assert newcombe(56, 70, 48, 80) == pytest.approx((-0.3339, -0.0524), abs=1e-4)
    lo, hi = newcombe(5, 10, 5, 10)
    assert lo < 0 < hi and lo == pytest.approx(-hi)
    assert newcombe(0, 0, 1, 2) is None


def test_quantile_linear():
    xs = [1.0, 2.0, 3.0, 4.0]
    assert quantile(xs, 0.5) == 2.5 and quantile(xs, 0.95) == pytest.approx(3.85)
    assert quantile([7.0], 0.025) == 7.0


def test_bootstrap_is_deterministic_with_seed():
    a = [1.0, 2.0, 3.0, 4.0, 5.0, 3.0, 2.0, 4.0]
    b = [3.0, 4.0, 5.0, 4.0, 5.0, 4.0, 3.0, 5.0]
    r1 = bootstrap_means(a, b, resamples=500, seed=7)
    r2 = bootstrap_means(a, b, resamples=500, seed=7)
    r3 = bootstrap_means(a, b, resamples=500, seed=8)
    assert r1 == r2 and r1 != r3
    assert r1.a_ci[0] <= sum(a) / len(a) <= r1.a_ci[1]
    assert r1.b_ci[0] <= sum(b) / len(b) <= r1.b_ci[1]
    assert r1.delta_ci[0] > 0  # b is clearly higher
    assert excludes_zero(r1.delta_ci) is True


def test_bootstrap_edge_cases():
    same = bootstrap_means([2.0] * 5, [2.0] * 5, resamples=50, seed=1)
    assert same.delta_ci == (0.0, 0.0) and excludes_zero(same.delta_ci) is False
    empty = bootstrap_means([], [1.0, 2.0], resamples=50, seed=1)
    assert empty.a_ci is None and empty.delta_ci is None and empty.b_ci is not None
    with pytest.raises(ValueError):
        bootstrap_means([1.0], [1.0], resamples=0)


def test_excludes_zero():
    assert excludes_zero((0.1, 0.3)) and excludes_zero((-0.3, -0.01))
    assert excludes_zero((-0.1, 0.2)) is False and excludes_zero((0.0, 0.2)) is False
    assert excludes_zero(None) is None

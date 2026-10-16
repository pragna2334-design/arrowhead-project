"""Percentiles (linear interpolation, numpy's default) and per-metric summaries."""

from __future__ import annotations

import math
from collections.abc import Iterable, Sequence

from bench.models import METRICS, BenchSample, MetricStats


def percentile(values: Sequence[float], p: float) -> float | None:
    """p in [0, 100]; linear interpolation between closest ranks. None for no values."""
    xs = sorted(values)
    if not xs:
        return None
    if not 0 <= p <= 100:
        raise ValueError("p must be in [0, 100]")
    k = (len(xs) - 1) * p / 100
    lo, hi = math.floor(k), math.ceil(k)
    return xs[lo] + (xs[hi] - xs[lo]) * (k - lo)


def _r(x: float | None) -> float | None:
    return None if x is None else round(x, 1)


def summarize(values: Iterable[float]) -> MetricStats:
    xs = [float(v) for v in values]
    if not xs:
        return MetricStats(n=0)
    return MetricStats(
        n=len(xs),
        p50=_r(percentile(xs, 50)),
        p95=_r(percentile(xs, 95)),
        mean=_r(sum(xs) / len(xs)),
        min=_r(min(xs)),
        max=_r(max(xs)),
    )


def stats_for(samples: Sequence[BenchSample]) -> dict[str, MetricStats]:
    """Summaries over successful samples; a metric that is None in a sample is skipped."""
    ok = [s for s in samples if s.error is None]
    out: dict[str, MetricStats] = {}
    for m in METRICS:
        out[m] = summarize(v for s in ok if (v := getattr(s, m)) is not None)
    return out

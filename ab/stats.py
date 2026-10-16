"""Small stdlib statistics for A/B comparisons (no numpy).

- Proportions: Wilson score interval per arm; Newcombe's hybrid score interval (his method 10)
  for the difference b - a. Both behave well for small n and for p near 0 or 1, where the
  textbook Wald interval collapses.
- Means: percentile bootstrap per arm and for the difference of means (each arm resampled
  independently, `resamples` times, seeded so results are reproducible).

Caveats (shown in the report): calls are treated as independent samples, but repeats of the
same persona are correlated, so intervals are somewhat optimistic; no correction is made for
testing many metrics at once (at 95%, about 1 in 20 null differences looks "significant").
"""

from __future__ import annotations

import math
import random
from collections.abc import Sequence
from dataclasses import dataclass
from statistics import NormalDist

Interval = tuple[float, float]


def z_for(confidence: float = 0.95) -> float:
    """Two-sided normal quantile, e.g. 1.959964 for 95%."""
    if not 0 < confidence < 1:
        raise ValueError(f"confidence must be in (0, 1), got {confidence}")
    return NormalDist().inv_cdf(1 - (1 - confidence) / 2)


def wilson(successes: int, n: int, confidence: float = 0.95) -> Interval | None:
    """Wilson score interval for successes/n. None when n == 0."""
    if n <= 0:
        return None
    if not 0 <= successes <= n:
        raise ValueError(f"successes {successes} not in [0, {n}]")
    z = z_for(confidence)
    p = successes / n
    z2n = z * z / n
    centre = (p + z2n / 2) / (1 + z2n)
    half = z / (1 + z2n) * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n))
    lo = 0.0 if successes == 0 else max(0.0, centre - half)  # exact at the boundaries
    hi = 1.0 if successes == n else min(1.0, centre + half)
    return lo, hi


def newcombe(x_a: int, n_a: int, x_b: int, n_b: int, confidence: float = 0.95) -> Interval | None:
    """Newcombe hybrid score interval for p_b - p_a (method 10, Newcombe 1998).

    L = d - sqrt((p_b - l_b)^2 + (u_a - p_a)^2),  U = d + sqrt((u_b - p_b)^2 + (p_a - l_a)^2)
    where (l, u) are the Wilson limits of each arm. None when either arm is empty."""
    wa, wb = wilson(x_a, n_a, confidence), wilson(x_b, n_b, confidence)
    if wa is None or wb is None:
        return None
    pa, pb = x_a / n_a, x_b / n_b
    (la, ua), (lb, ub) = wa, wb
    d = pb - pa
    lo = d - math.sqrt((pb - lb) ** 2 + (ua - pa) ** 2)
    hi = d + math.sqrt((ub - pb) ** 2 + (pa - la) ** 2)
    return max(-1.0, lo), min(1.0, hi)


def quantile(sorted_xs: Sequence[float], q: float) -> float:
    """Linear-interpolation quantile (numpy's default, type 7) of an already sorted list."""
    if not sorted_xs:
        raise ValueError("quantile of an empty sequence")
    rank = q * (len(sorted_xs) - 1)
    lo = math.floor(rank)
    hi = min(lo + 1, len(sorted_xs) - 1)
    return sorted_xs[lo] + (sorted_xs[hi] - sorted_xs[lo]) * (rank - lo)


@dataclass(frozen=True)
class BootstrapResult:
    a_ci: Interval | None
    b_ci: Interval | None
    delta_ci: Interval | None
    resamples: int


def _mean(xs: Sequence[float]) -> float:
    return sum(xs) / len(xs)


def bootstrap_means(
    a: Sequence[float],
    b: Sequence[float],
    *,
    resamples: int = 2000,
    seed: int | str = 7,
    confidence: float = 0.95,
) -> BootstrapResult:
    """Percentile bootstrap CIs for mean(a), mean(b) and mean(b) - mean(a).

    Each arm is resampled with replacement at its own size, independently; the same seed always
    gives the same intervals. An empty arm gets no CI (and the delta gets none)."""
    if resamples < 1:
        raise ValueError("resamples must be >= 1")
    rng = random.Random(seed)
    xa, xb = list(a), list(b)
    ma: list[float] = []
    mb: list[float] = []
    for _ in range(resamples):
        if xa:
            ma.append(_mean(rng.choices(xa, k=len(xa))))
        if xb:
            mb.append(_mean(rng.choices(xb, k=len(xb))))
    alpha = (1 - confidence) / 2

    def ci(vals: list[float]) -> Interval | None:
        if not vals:
            return None
        s = sorted(vals)
        return quantile(s, alpha), quantile(s, 1 - alpha)

    delta = [y - x for x, y in zip(ma, mb, strict=True)] if xa and xb else []
    return BootstrapResult(ci(ma), ci(mb), ci(delta), resamples)


def excludes_zero(ci: Interval | None) -> bool | None:
    """True when the interval lies strictly on one side of 0; None when there is no interval."""
    if ci is None:
        return None
    return ci[0] > 0 or ci[1] < 0

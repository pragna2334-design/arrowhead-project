"""Compare two bench runs: python -m bench.compare BEFORE.json AFTER.json

Prints p50/p95 per metric with deltas (after - before; negative = faster).
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from pydantic import ValidationError

from bench.models import METRICS, BenchReport

_META_FIELDS = ("label", "model", "thinking", "prompt_version", "scenario", "aggregation")


def load_report(path: str | Path) -> BenchReport:
    return BenchReport.model_validate_json(Path(path).read_text())


def _fmt(x: float | None) -> str:
    return "-" if x is None else f"{x:.0f}"


def _delta(a: float | None, b: float | None) -> str:
    if a is None or b is None:
        return "-"
    d = b - a
    pct = f" ({d / a * 100:+.0f}%)" if a else ""
    return f"{d:+.0f}{pct}"


def render(before: BenchReport, after: BenchReport) -> str:
    lines = []
    w = max(len("setting"), *(len(f) for f in _META_FIELDS))
    lines.append(f"{'setting':<{w}}  {'before':<20}  after")
    for f in _META_FIELDS:
        a, b = getattr(before.meta, f), getattr(after.meta, f)
        mark = "" if a == b else "  *"
        lines.append(f"{f:<{w}}  {a!s:<20}  {b}{mark}")
    lines.append("")
    header = (
        f"{'metric':<20}{'p50 before':>11}{'p50 after':>11}{'Δ p50':>14}"
        f"{'p95 before':>12}{'p95 after':>11}{'Δ p95':>14}{'n':>9}"
    )
    lines.append(header)
    lines.append("-" * len(header))
    for m in METRICS:
        sa, sb = before.stats.get(m), after.stats.get(m)
        a50, b50 = (sa.p50 if sa else None), (sb.p50 if sb else None)
        a95, b95 = (sa.p95 if sa else None), (sb.p95 if sb else None)
        n = f"{sa.n if sa else 0}/{sb.n if sb else 0}"
        lines.append(
            f"{m:<20}{_fmt(a50):>11}{_fmt(b50):>11}{_delta(a50, b50):>14}"
            f"{_fmt(a95):>12}{_fmt(b95):>11}{_delta(a95, b95):>14}{n:>9}"
        )
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(prog="python -m bench.compare", description=__doc__.split("\n")[0])
    p.add_argument("before")
    p.add_argument("after")
    args = p.parse_args(argv)
    try:
        before, after = load_report(args.before), load_report(args.after)
    except (OSError, ValidationError, ValueError) as e:
        print(f"error: {e}", file=sys.stderr)
        return 2
    print(render(before, after))
    return 0


if __name__ == "__main__":
    sys.exit(main())

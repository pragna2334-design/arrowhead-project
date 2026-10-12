"""Small stdlib statistics helpers for the report (no numpy)."""

from __future__ import annotations

import json
import sys
from dataclasses import dataclass
from pathlib import Path

from schemas import LatencyRecord

LATENCY_FIELDS = ("e2e_ms", "pipeline_ms", "stt_ms", "llm_ttfb_ms", "tts_ttfb_ms")


def percentile(values: list[float], p: float) -> float | None:
    """Percentile with linear interpolation between closest ranks.

    Same as numpy's default ("linear", Hyndman & Fan type 7): sort the values, take the
    fractional rank r = p/100 * (n - 1), and interpolate between the values at floor(r) and
    ceil(r). Returns None for an empty list. p must be in [0, 100].
    """
    if not 0 <= p <= 100:
        raise ValueError(f"percentile p must be in [0, 100], got {p}")
    xs = sorted(values)
    if not xs:
        return None
    rank = p / 100 * (len(xs) - 1)
    lo = int(rank)
    hi = min(lo + 1, len(xs) - 1)
    return xs[lo] + (xs[hi] - xs[lo]) * (rank - lo)


@dataclass(frozen=True)
class LatencyStat:
    field: str
    n: int
    p50: float | None
    p95: float | None


@dataclass(frozen=True)
class LatencySummary:
    n_turns: int  # bot turns with kind == "turn"
    n_greetings_excluded: int
    n_calls: int
    stats: dict[str, LatencyStat]


def load_latency(path: Path | None, version: str | None = None) -> list[LatencyRecord]:
    """Read a latency JSONL file. Missing file -> []. Blank and malformed lines are skipped
    (a crashed voice process can leave a truncated last line). If `version` is given, only
    records for that prompt version (or with no version) are kept."""
    if path is None or not path.exists():
        return []
    records = []
    for i, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        try:
            rec = LatencyRecord.model_validate(json.loads(line))
        except ValueError as exc:
            print(f"warning: {path}:{i}: skipping invalid latency record: {exc}", file=sys.stderr)
            continue
        if version is None or rec.prompt_version in (None, version):
            records.append(rec)
    return records


def summarize_latency(records: list[LatencyRecord]) -> LatencySummary | None:
    """p50/p95 per field over bot turns only; greetings are excluded. None if no turns."""
    turns = [r for r in records if r.kind == "turn"]
    if not turns:
        return None
    stats = {}
    for f in LATENCY_FIELDS:
        vals = [v for r in turns if (v := getattr(r, f)) is not None]
        stats[f] = LatencyStat(f, len(vals), percentile(vals, 50), percentile(vals, 95))
    return LatencySummary(
        n_turns=len(turns),
        n_greetings_excluded=sum(r.kind == "greeting" for r in records),
        n_calls=len({r.call_id for r in turns}),
        stats=stats,
    )

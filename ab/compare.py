"""Prompt-version A/B comparison: two Summary files -> one Comparison file.

python -m ab.compare --a v1 --b v2          (reads results/<v>/summary.json)
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

from schemas import Comparison, MetricDelta, PersonaDiff, ResultsLayout, Summary

OUTCOMES = ("callback_booked", "declined", "dnc", "incomplete")


def _d(a: float, b: float) -> MetricDelta:
    return MetricDelta(a=round(a, 4), b=round(b, 4), delta=round(b - a, 4))


def _share(s: Summary, outcome: str) -> float:
    return s.outcomes.get(outcome, 0) / s.n_calls if s.n_calls else 0.0


def _red_team_rate(s: Summary) -> float:
    return sum(r.passed for r in s.red_team) / len(s.red_team) if s.red_team else 0.0


def compare(a: Summary, b: Summary) -> Comparison:
    rules = sorted(set(a.violation_counts) | set(b.violation_counts))
    dims = sorted(set(a.judge_means) | set(b.judge_means))
    a_by = {c.persona_id: c for c in a.calls if c.persona_id}
    b_by = {c.persona_id: c for c in b.calls if c.persona_id}
    a_rows = {r.persona_id: r for r in a.per_persona}
    b_rows = {r.persona_id: r for r in b.per_persona}
    personas = sorted(set(a_by) | set(b_by))
    return Comparison(
        a=a.prompt_version,
        b=b.prompt_version,
        generated_at=time.time(),
        violation_rate=_d(a.violation_rate, b.violation_rate),
        consent_rate=_d(a.consent_rate, b.consent_rate),
        outcomes={o: _d(_share(a, o), _share(b, o)) for o in OUTCOMES},
        violation_counts={
            r: _d(a.violation_counts.get(r, 0), b.violation_counts.get(r, 0)) for r in rules
        },
        judge_means={k: _d(a.judge_means.get(k, 0.0), b.judge_means.get(k, 0.0)) for k in dims},
        red_team_pass_rate=_d(_red_team_rate(a), _red_team_rate(b)),
        per_persona=[
            PersonaDiff(
                persona_id=p,
                a_outcome=a_by[p].outcome if p in a_by else None,
                b_outcome=b_by[p].outcome if p in b_by else None,
                a_violations=a_rows[p].violations if p in a_rows else 0,
                b_violations=b_rows[p].violations if p in b_rows else 0,
                a_judge_mean=a_rows[p].judge_mean if p in a_rows else None,
                b_judge_mean=b_rows[p].judge_mean if p in b_rows else None,
            )
            for p in personas
        ],
        a_summary=a,
        b_summary=b,
    )


def main(argv: list[str] | None = None) -> Path:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--a", required=True)
    ap.add_argument("--b", required=True)
    ap.add_argument("--results", default="results")
    args = ap.parse_args(argv)
    root = Path(args.results)
    la, lb = ResultsLayout(root, args.a), ResultsLayout(root, args.b)
    for lay in (la, lb):
        if not lay.summary.exists():
            raise SystemExit(f"missing {lay.summary}; run `make eval PROMPT={lay.version}` first")
    sa = Summary.model_validate_json(la.summary.read_text())
    sb = Summary.model_validate_json(lb.summary.read_text())
    out = la.comparison(args.b)
    out.write_text(compare(sa, sb).model_dump_json(indent=2))
    print(f"wrote {out}")
    return out


if __name__ == "__main__":
    main()

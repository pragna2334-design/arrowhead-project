"""Prompt-version A/B comparison: two Summary files -> one Comparison file, with uncertainty.

    python -m ab.compare --a v1 --b v2 [--scenario ID] [--results DIR]

Reads results/<scenario>/<v>/summary.json for both versions and writes
results/<scenario>/compare_<a>_<b>.json. Scenario defaults to config/env (SCENARIO).

Every rate and mean is computed from the per-call samples in Summary.calls (several repeats per
persona with `make eval REPEATS=3`):
- proportions (violation, consent, outcome shares, red-team pass, hallucinated figure): Wilson
  95% CI per arm and Newcombe's hybrid score CI for the difference;
- means (each judge dimension, production cost per call): percentile bootstrap CI per arm and
  for the difference (config compare.bootstrap_resamples, compare.seed).
`significant` is True only when the CI of the difference excludes 0. A non-significant delta
must never be presented as an improvement or a regression (the report shows "n.s.").
"""

from __future__ import annotations

import argparse
import sys
import time
from collections.abc import Callable
from pathlib import Path
from typing import Any

from ab.stats import bootstrap_means, excludes_zero, newcombe, wilson
from schemas import CallScore, Comparison, MetricDelta, PersonaDiff, ResultsLayout, Summary

OUTCOMES = ("callback_booked", "declined", "dnc", "incomplete")
RULES = (
    "disclosure",
    "dnc_honored",
    "rate_in_range",
    "no_guaranteed_approval",
    "callback_in_hours",
    "no_pii_leak",
    "grounded_figures",
    "scenario_claims",
)
MIN_N_PER_ARM = 30
DEFAULT_RESAMPLES = 2000
DEFAULT_SEED = 7
DEFAULT_CONFIDENCE = 0.95


def _r(x: float) -> float:
    return round(x, 4)


def _ci(ci: tuple[float, float] | None, digits: int = 4) -> tuple[float, float] | None:
    return None if ci is None else (round(ci[0], digits), round(ci[1], digits))


def _d(a: float, b: float) -> MetricDelta:
    """Plain delta without uncertainty (counts, or summaries without per-call samples)."""
    return MetricDelta(a=_r(a), b=_r(b), delta=_r(b - a))


def _share(s: Summary, outcome: str) -> float:
    return s.outcomes.get(outcome, 0) / s.n_calls if s.n_calls else 0.0


def _red_team_rate(s: Summary) -> float:
    return sum(r.passed for r in s.red_team) / len(s.red_team) if s.red_team else 0.0


# ---------------------------------------------------------------- per-call predicates


def has_violation(c: CallScore) -> bool:
    return bool(c.compliance_violations) or any(getattr(c.rules, r, None) is False for r in RULES)


def has_hallucinated_figure(c: CallScore) -> bool:
    return c.rules.grounded_figures is False or any(
        v.rule == "grounded_figures" for v in c.compliance_violations
    )


def proportion_delta(
    a: list[bool], b: list[bool], confidence: float = DEFAULT_CONFIDENCE
) -> MetricDelta:
    """Share of True per arm with Wilson CIs, delta with a Newcombe CI."""
    xa, na, xb, nb = sum(a), len(a), sum(b), len(b)
    pa = xa / na if na else 0.0
    pb = xb / nb if nb else 0.0
    dci = newcombe(xa, na, xb, nb, confidence)
    return MetricDelta(
        a=_r(pa),
        b=_r(pb),
        delta=_r(pb - pa),
        n_a=na,
        n_b=nb,
        a_ci95=_ci(wilson(xa, na, confidence)),
        b_ci95=_ci(wilson(xb, nb, confidence)),
        delta_ci95=_ci(dci),
        significant=excludes_zero(dci),
        method="wilson+newcombe",
    )


def mean_delta(
    a: list[float],
    b: list[float],
    *,
    resamples: int = DEFAULT_RESAMPLES,
    seed: int | str = DEFAULT_SEED,
    confidence: float = DEFAULT_CONFIDENCE,
    digits: int = 4,
) -> MetricDelta:
    """Mean per arm with percentile-bootstrap CIs for each arm and for the difference."""
    ma = sum(a) / len(a) if a else 0.0
    mb = sum(b) / len(b) if b else 0.0
    bs = bootstrap_means(a, b, resamples=resamples, seed=seed, confidence=confidence)
    return MetricDelta(
        a=round(ma, digits),
        b=round(mb, digits),
        delta=round(mb - ma, digits),
        n_a=len(a),
        n_b=len(b),
        a_ci95=_ci(bs.a_ci, digits),
        b_ci95=_ci(bs.b_ci, digits),
        delta_ci95=_ci(bs.delta_ci, digits),
        significant=excludes_zero(bs.delta_ci),
        method=f"bootstrap-{resamples}",
    )


# ---------------------------------------------------------------- per persona


def _repeat_key(call_id: str) -> tuple[str, int]:
    """Natural order of '<version>-<persona>-<repeat>' ids, so repeat 10 sorts after 2."""
    head, _, tail = call_id.rpartition("-")
    return (head, int(tail)) if tail.isdigit() else (call_id, 0)


def _modal_outcomes(s: Summary) -> dict[str, str]:
    """Per persona: the most common outcome across repeats (ties -> earliest repeat).
    Violation counts in the diff are summed over all repeats (PersonaRow.violations)."""
    by: dict[str, list[str]] = {}
    for c in sorted(s.calls, key=lambda c: _repeat_key(c.call_id)):
        if c.persona_id:
            by.setdefault(c.persona_id, []).append(c.outcome)
    return {p: max(outs, key=lambda o: (outs.count(o), -outs.index(o))) for p, outs in by.items()}


def repeats_per_persona(*summaries: Summary) -> int | None:
    """Most calls seen for any single persona in either summary (None without per-call data)."""
    counts: dict[tuple[int, str], int] = {}
    for i, s in enumerate(summaries):
        for c in s.calls:
            key = (i, c.persona_id or c.call_id)
            counts[key] = counts.get(key, 0) + 1
    return max(counts.values()) if counts else None


# ---------------------------------------------------------------- notes


def _label_value(label: str, m: MetricDelta, kind: str) -> str:
    if kind == "pct":
        return f"{label} {m.delta * 100:+.1f} pp"
    if kind == "usd":
        return f"{label} {m.delta:+.4f} USD"
    return f"{label} {m.delta:+.2f}"


def build_notes(
    a: Summary,
    b: Summary,
    tested: list[tuple[str, MetricDelta, str]],
    confidence: float,
) -> list[str]:
    notes: list[str] = []
    level = f"{confidence * 100:g}%"
    if not a.calls or not b.calls:
        notes.append(
            "No per-call data in at least one summary: confidence intervals could not be "
            "computed, so no difference can be called significant."
        )
    if a.scenario != b.scenario:
        notes.append(
            f"A is scenario {a.scenario!r} and B is {b.scenario!r}: the comparison mixes use cases."
        )
    small = [(name, s.n_calls) for name, s in (("A", a), ("B", b)) if s.n_calls < MIN_N_PER_ARM]
    if small:
        sizes = " and ".join(f"{n} calls in {name}" for name, n in small)
        notes.append(
            f"Small sample: {sizes} (< {MIN_N_PER_ARM} per arm). Intervals are wide; run more "
            "repeats (make eval REPEATS=3) before drawing conclusions."
        )
    computed = [(lbl, m, k) for lbl, m, k in tested if m.significant is not None]
    ns = [lbl for lbl, m, _ in computed if not m.significant]
    sig = [_label_value(lbl, m, k) for lbl, m, k in computed if m.significant]
    if computed:
        notes.append(
            f"{len(ns)} of {len(computed)} differences are not statistically significant at "
            f"{level} (marked n.s.); treat them as noise."
        )
        if sig:
            notes.append(
                f"Significant at {level} (CI of B − A excludes 0): " + "; ".join(sig) + "."
            )
        notes.append(
            f"No multiple-comparison correction: with {len(computed)} metrics, about "
            f"{len(computed) * (1 - confidence):.1f} could look significant by chance. Repeats of "
            "the same persona are correlated, so intervals are somewhat optimistic."
        )
    return notes


# ---------------------------------------------------------------- compare


def compare(
    a: Summary,
    b: Summary,
    *,
    resamples: int = DEFAULT_RESAMPLES,
    seed: int = DEFAULT_SEED,
    confidence: float = DEFAULT_CONFIDENCE,
) -> Comparison:
    rules = sorted(set(a.violation_counts) | set(b.violation_counts))
    dims = sorted(set(a.judge_means) | set(b.judge_means))
    a_by, b_by = _modal_outcomes(a), _modal_outcomes(b)
    a_rows = {r.persona_id: r for r in a.per_persona}
    b_rows = {r.persona_id: r for r in b.per_persona}
    personas = sorted(set(a_by) | set(b_by))
    samples = bool(a.calls) and bool(b.calls)

    def prop(pred: Callable[[CallScore], bool], fallback: tuple[float, float]) -> MetricDelta:
        if not samples:
            return _d(*fallback)
        return proportion_delta([pred(c) for c in a.calls], [pred(c) for c in b.calls], confidence)

    def boot(name: str, get: Callable[[CallScore], float | None], fallback, digits=4):
        va = [v for c in a.calls if (v := get(c)) is not None]
        vb = [v for c in b.calls if (v := get(c)) is not None]
        if not samples or not va or not vb:
            return _d(*fallback) if fallback is not None else None
        # seed per metric: results do not depend on the order metrics are computed in
        return mean_delta(
            va, vb, resamples=resamples, seed=f"{seed}-{name}", confidence=confidence, digits=digits
        )

    violation_rate = prop(has_violation, (a.violation_rate, b.violation_rate))
    consent_rate = prop(lambda c: c.consent_captured, (a.consent_rate, b.consent_rate))
    outcomes = {
        o: prop(lambda c, o=o: c.outcome == o, (_share(a, o), _share(b, o))) for o in OUTCOMES
    }
    if samples:
        ra = [bool(c.red_team_passed) for c in a.calls if c.red_team_passed is not None]
        rb = [bool(c.red_team_passed) for c in b.calls if c.red_team_passed is not None]
        red_team = (
            proportion_delta(ra, rb, confidence)
            if ra and rb
            else _d(_red_team_rate(a), _red_team_rate(b))
        )
        halluc = proportion_delta(
            [has_hallucinated_figure(c) for c in a.calls],
            [has_hallucinated_figure(c) for c in b.calls],
            confidence,
        )
    else:
        red_team = _d(_red_team_rate(a), _red_team_rate(b))
        halluc = None
    judge_means = {
        k: boot(
            f"judge-{k}",
            lambda c, k=k: getattr(c.judge, k).score if c.judge is not None else None,
            (a.judge_means.get(k, 0.0), b.judge_means.get(k, 0.0)),
        )
        for k in dims
    }
    cost = boot(
        "cost",
        lambda c: c.cost.production_usd if c.cost is not None else None,
        (a.cost.production_usd_mean, b.cost.production_usd_mean) if a.cost and b.cost else None,
        digits=6,
    )

    tested: list[tuple[str, MetricDelta, str]] = [
        ("violation rate", violation_rate, "pct"),
        ("consent rate", consent_rate, "pct"),
        ("red-team pass rate", red_team, "pct"),
        *((f"{o} share", m, "pct") for o, m in outcomes.items()),
        *((f"judge {k.replace('_', ' ')}", m, "score") for k, m in judge_means.items()),
    ]
    if halluc is not None:
        tested.append(("hallucinated-figure rate", halluc, "pct"))
    if cost is not None:
        tested.append(("production cost per call", cost, "usd"))

    return Comparison(
        a=a.prompt_version,
        b=b.prompt_version,
        generated_at=time.time(),
        violation_rate=violation_rate,
        consent_rate=consent_rate,
        outcomes=outcomes,
        violation_counts={
            r: _d(a.violation_counts.get(r, 0), b.violation_counts.get(r, 0)) for r in rules
        },
        judge_means=judge_means,
        red_team_pass_rate=red_team,
        per_persona=[
            PersonaDiff(
                persona_id=p,
                a_outcome=a_by.get(p),
                b_outcome=b_by.get(p),
                a_violations=a_rows[p].violations if p in a_rows else 0,
                b_violations=b_rows[p].violations if p in b_rows else 0,
                a_judge_mean=a_rows[p].judge_mean if p in a_rows else None,
                b_judge_mean=b_rows[p].judge_mean if p in b_rows else None,
            )
            for p in personas
        ],
        a_summary=a,
        b_summary=b,
        scenario=a.scenario,
        repeats_per_persona=repeats_per_persona(a, b),
        cost_per_call=cost,
        hallucinated_figure_rate=halluc,
        notes=build_notes(a, b, tested, confidence),
    )


# ---------------------------------------------------------------- CLI


def _config(scenario: str | None) -> Any | None:
    try:
        from brain.config import load_config

        return load_config(scenario=scenario)
    except Exception as exc:  # noqa: BLE001 - compare must work without a loadable config
        print(f"warning: config not loaded ({exc}); using defaults", file=sys.stderr)
        return None


def main(argv: list[str] | None = None) -> Path:
    ap = argparse.ArgumentParser(prog="python -m ab.compare", description=__doc__.split("\n\n")[0])
    ap.add_argument("--a", required=True, help="baseline prompt version")
    ap.add_argument("--b", required=True, help="candidate prompt version")
    ap.add_argument("--scenario", default=None, help="scenario id (default: config/env SCENARIO)")
    ap.add_argument("--results", default=None, help="results root (default: config paths)")
    args = ap.parse_args(argv)
    cfg = _config(args.scenario)
    scenario = args.scenario or (cfg.scenario.id if cfg else "personal_loan")
    cmp_cfg = (cfg.compare if cfg else {}) or {}
    root = Path(args.results or (cfg.results_dir if cfg else "results"))
    la = ResultsLayout(root, args.a, scenario=scenario)
    lb = ResultsLayout(root, args.b, scenario=scenario)
    for lay in (la, lb):
        if not lay.summary.exists():
            raise SystemExit(
                f"missing {lay.summary}; run `make eval PROMPT={lay.version} "
                f"SCENARIO={scenario}` first"
            )
    sa = Summary.model_validate_json(la.summary.read_text())
    sb = Summary.model_validate_json(lb.summary.read_text())
    comp = compare(
        sa,
        sb,
        resamples=int(cmp_cfg.get("bootstrap_resamples", DEFAULT_RESAMPLES)),
        seed=int(cmp_cfg.get("seed", DEFAULT_SEED)),
        confidence=float(cmp_cfg.get("confidence", DEFAULT_CONFIDENCE)),
    )
    out = la.comparison(args.b)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(comp.model_dump_json(indent=2))
    print(f"wrote {out}")
    for note in comp.notes:
        print(f"note: {note}")
    return out


if __name__ == "__main__":
    main()

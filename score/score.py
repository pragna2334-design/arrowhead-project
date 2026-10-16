"""Score every transcript of one (scenario, prompt version) and aggregate a Summary.

    python -m score.score [--scenario ID] [--version v1] [--results results] [--no-judge]
                          [--fresh]

Reads results/<scenario>/<v>/transcripts/*.json, writes results/<scenario>/<v>/scores/
<call_id>.json (resumable: existing scores are kept unless --fresh; with the judge enabled,
scores whose judge previously failed are re-scored) and results/<scenario>/<v>/costs/
<call_id>.json (always re-priced from the current price table), then aggregates ALL score
files into results/<scenario>/<v>/summary.json. Scenario and version default to config/env
(SCENARIO, PROMPT_VERSION).

Rules: the six deterministic rules (score/rules.py, parameters from the scenario pack) plus
grounded_figures (score/grounding.py) and scenario_claims (score/scenario_rules.py).
"""

from __future__ import annotations

import argparse
import asyncio
import os
import re
import sys
import time
from collections import Counter
from collections.abc import Iterable
from pathlib import Path
from statistics import mean
from typing import get_args

from schemas import (
    CallScore,
    CostRecord,
    HallucinatedFigureRow,
    PersonaRow,
    RedTeamRow,
    ResultsLayout,
    ScenarioConfig,
    Summary,
    Transcript,
)
from schemas.common import Outcome, RuleName
from score import grounding, scenario_rules
from score.cost import cost_record, has_usage, summarize_costs
from score.judge import DIMS, JudgeResult, make_judge
from score.personas import PersonaInfo, persona_info_for
from score.rules import (
    RuleParams,
    consent_captured,
    describe_red_team,
    evaluate_red_team,
    evaluate_rules,
    objection_handled,
    outcome_of,
)

OUTCOMES: tuple[str, ...] = get_args(Outcome)
RULES: tuple[str, ...] = get_args(RuleName)


def repeat_of(call_id: str) -> int:
    """Repeat index from a '<version>-<persona>-<n>' call id; 0 when there is no int suffix."""
    head, sep, tail = call_id.rpartition("-")
    return int(tail) if sep and head and tail.isdigit() else 0


def _scenario_checks(t: Transcript, scenario: ScenarioConfig | None):
    """grounded_figures + scenario_claims. Called through the modules (not imported names) so
    the real implementations and test monkeypatches are picked up."""
    if scenario is None:
        return None, [], None, []
    g_ok, g_v = grounding.check(t, scenario)
    s_ok, s_v = scenario_rules.check(t, scenario)
    return g_ok, list(g_v), s_ok, list(s_v)


def score_transcript(
    t: Transcript,
    params: RuleParams,
    *,
    persona: PersonaInfo | None = None,
    judge: JudgeResult | None = None,
    judge_model: str | None = None,
    scenario: ScenarioConfig | None = None,
    cost: CostRecord | None = None,
) -> CallScore:
    """Pure: rules (+ scenario rules when `scenario` is given) + (optional, precomputed) judge
    result -> CallScore."""
    persona = persona or PersonaInfo(t.persona_id)
    rules, violations = evaluate_rules(t, params)
    g_ok, g_v, s_ok, s_v = _scenario_checks(t, scenario)
    if scenario is not None:
        rules = rules.model_copy(update={"grounded_figures": g_ok, "scenario_claims": s_ok})
        violations = sorted(
            violations + g_v + s_v,
            key=lambda v: (v.turn_index if v.turn_index is not None else -1, v.rule),
        )
    scores = judge.scores if judge else None
    red_passed, red_reason = None, None
    if persona.red_team and persona.attack_type:
        red_passed, red_reason = evaluate_red_team(
            persona.attack_type, rules, violations, scores, judge.held_rules if judge else None
        )
        if judge and judge.held_rules_reason:
            red_reason = f"{red_reason}; judge: {judge.held_rules_reason}"
    elif persona.red_team:  # unknown attack type: pass only with zero violations
        red_passed = not violations
        red_reason = "no violations" if red_passed else "rule violations under attack"
    return CallScore(
        call_id=t.call_id,
        persona_id=t.persona_id,
        prompt_version=t.prompt_version,
        scenario=scenario.id if scenario is not None else t.scenario,
        repeat=repeat_of(t.call_id),
        turns=len(t.turns),
        outcome=outcome_of(t),
        disclosure_ok=rules.disclosure,
        consent_captured=consent_captured(t),
        objection_handled=objection_handled(t, violations, scores),
        rules=rules,
        compliance_violations=violations,
        judge=scores,
        judge_model=judge_model if judge is not None and judge.scores is not None else None,
        judge_error=judge.error if judge else None,
        judge_usage=judge.usage if judge else None,
        cost=cost,
        is_red_team=persona.red_team,
        attack_type=persona.attack_type,
        red_team_passed=red_passed,
        red_team_reason=red_reason,
    )


def _has_violation(s: CallScore) -> bool:
    return bool(s.compliance_violations) or any(getattr(s.rules, r) is False for r in RULES)


def _judge_mean(s: CallScore) -> float | None:
    if s.judge is None:
        return None
    return mean(d.score for d in s.judge.as_dict().values())


_FIGURE_RE = re.compile(
    r"hallucinated figure:?\s*(.+?)(?:\s+not\s+in\b|\s*$)", re.IGNORECASE | re.DOTALL
)


def figure_from_detail(detail: str) -> str:
    """'hallucinated figure: 9.5% not in ...' -> '9.5%' (whole detail if the shape differs)."""
    m = _FIGURE_RE.search(detail or "")
    fig = m.group(1).strip().strip("'\"") if m else ""
    return fig or (detail or "").strip()


def hallucinated_figures(calls: Iterable[CallScore]) -> list[HallucinatedFigureRow]:
    return [
        HallucinatedFigureRow(
            call_id=s.call_id,
            persona_id=s.persona_id,
            turn_index=v.turn_index,
            figure=figure_from_detail(v.detail),
            excerpt=v.excerpt,
        )
        for s in calls
        for v in s.compliance_violations
        if v.rule == "grounded_figures"
    ]


def summarize(
    scores: Iterable[CallScore],
    *,
    prompt_version: str,
    agent_model: str,
    judge_model: str | None,
    generated_at: float | None = None,
    scenario: str | None = None,
) -> Summary:
    calls = sorted(scores, key=lambda s: s.call_id)
    if scenario is None:  # take it from the calls (default scenario if there are none)
        scenario = calls[0].scenario if calls else "personal_loan"
    n = len(calls)
    outcomes = {o: sum(1 for s in calls if s.outcome == o) for o in OUTCOMES}
    vcounts = Counter(v.rule for s in calls for v in s.compliance_violations)
    for s in calls:  # a failed rule with no Violation object still counts once
        for r in RULES:
            if getattr(s.rules, r) is False and not any(
                v.rule == r for v in s.compliance_violations
            ):
                vcounts[r] += 1
    with_v = sum(1 for s in calls if _has_violation(s))
    judged = [s.judge for s in calls if s.judge is not None]
    means: dict[str, float] = {}
    dists: dict[str, dict[str, int]] = {}
    if judged:
        for d in DIMS:
            vals = [getattr(j, d).score for j in judged]
            means[d] = round(mean(vals), 2)
            dists[d] = {str(k): vals.count(k) for k in range(1, 6)}

    per_persona: list[PersonaRow] = []
    for pid in sorted({s.persona_id or "(none)" for s in calls}):
        group = [s for s in calls if (s.persona_id or "(none)") == pid]
        jm = [m for m in (_judge_mean(s) for s in group) if m is not None]
        per_persona.append(
            PersonaRow(
                persona_id=pid,
                calls=len(group),
                outcomes=dict(Counter(s.outcome for s in group)),
                violations=sum(len(s.compliance_violations) for s in group),
                judge_mean=round(mean(jm), 2) if jm else None,
            )
        )
    red_team = [
        RedTeamRow(
            persona_id=s.persona_id or "(none)",
            attack_type=s.attack_type,
            call_id=s.call_id,
            passed=bool(s.red_team_passed),
            reason=s.red_team_reason or describe_red_team(s),
        )
        for s in calls
        if s.is_red_team and s.attack_type is not None and s.red_team_passed is not None
    ]
    return Summary(
        prompt_version=prompt_version,
        scenario=scenario,
        generated_at=generated_at if generated_at is not None else time.time(),
        agent_model=agent_model,
        judge_model=judge_model,
        n_calls=n,
        outcomes=outcomes,
        violation_counts={r: vcounts.get(r, 0) for r in RULES},
        calls_with_violations=with_v,
        violation_rate=round(with_v / n, 3) if n else 0.0,
        consent_rate=round(sum(s.consent_captured for s in calls) / n, 3) if n else 0.0,
        judge_means=means,
        judge_distributions=dists,
        per_persona=per_persona,
        red_team=red_team,
        calls=calls,
        cost=summarize_costs(calls),
        hallucinated_figures=hallucinated_figures(calls),
    )


# ---------------------------------------------------------------- runner


def _needs_scoring(path: Path, transcript: Path, *, fresh: bool, use_judge: bool) -> bool:
    if fresh or not path.exists():
        return True
    if transcript.stat().st_mtime > path.stat().st_mtime:
        return True  # call was re-simulated after it was scored
    try:
        old = CallScore.model_validate_json(path.read_text())
    except ValueError:
        return True
    # With the judge available, (re)judge anything without judge scores: failed judge
    # calls and earlier --no-judge runs alike.
    return use_judge and old.judge is None


async def run(
    version: str | None = None,
    results: str | None = None,
    *,
    use_judge: bool = True,
    fresh: bool = False,
    config_path: str | None = None,
    scenario: str | None = None,
) -> Summary:
    from brain.config import load_config

    cfg = load_config(config_path, prompt_version=version, scenario=scenario)
    version = cfg.brain.prompt_version
    sc = cfg.scenario
    layout = ResultsLayout(Path(results or cfg.results_dir), version, scenario=sc.id)
    layout.scores.mkdir(parents=True, exist_ok=True)
    layout.costs.mkdir(parents=True, exist_ok=True)
    params = RuleParams.from_scenario(sc)
    persona_cache: dict = {}  # scenario -> personas.yaml map, read once

    judge = None
    judge_unavailable: str | None = None
    if use_judge:
        if os.getenv("DEEPSEEK_API_KEY"):
            judge = make_judge(cfg)
        else:
            judge_unavailable = "DEEPSEEK_API_KEY not set"
            print("warning: DEEPSEEK_API_KEY not set; scoring without the judge", file=sys.stderr)

    paths = sorted(layout.transcripts.glob("*.json")) if layout.transcripts.exists() else []
    if not paths:  # don't write an empty summary that would mask the real problem in the report
        print(
            f"error: no transcripts in {layout.transcripts}; run `make sim "
            f"SCENARIO={sc.id} PROMPT={version}` first",
            file=sys.stderr,
        )
        raise SystemExit(2)
    agent_models: set[str] = set()
    valid_ids: set[str] = set()
    errored: list[str] = []
    no_usage = 0
    for i, p in enumerate(paths, 1):
        t = Transcript.model_validate_json(p.read_text())
        out = layout.scores / f"{t.call_id}.json"
        cost_path = layout.costs / f"{t.call_id}.json"
        if t.stop_reason == "error":
            # A call that crashed (bad key, network) says nothing about the agent: never score
            # it, or it would count as a disclosure failure / an easy red-team "pass".
            errored.append(t.call_id)
            out.unlink(missing_ok=True)
            cost_path.unlink(missing_ok=True)
            continue
        if t.scenario != sc.id:
            print(
                f"warning: {t.call_id} was simulated for scenario {t.scenario!r}, scored as "
                f"{sc.id!r}",
                file=sys.stderr,
            )
        agent_models.add(t.agent_model)
        valid_ids.add(t.call_id)
        if not _needs_scoring(out, p, fresh=fresh, use_judge=judge is not None):
            # Kept score: only re-price it (prices may have changed; no LLM call needed).
            old = CallScore.model_validate_json(out.read_text())
            cost = cost_record(t, old.judge_usage, cfg)
            no_usage += not has_usage(cost)
            cost_path.write_text(cost.model_dump_json(indent=2))
            if old.cost != cost:
                out.write_text(old.model_copy(update={"cost": cost}).model_dump_json(indent=2))
            continue
        info = persona_info_for(t, persona_cache)
        jres = None
        if judge is not None:
            jres = await judge.score(
                t,
                persona_description=info.description,
                attack_type=info.attack_type if info.red_team else None,
            )
        elif judge_unavailable:
            jres = JudgeResult(error=judge_unavailable)
        cost = cost_record(t, jres.usage if jres else None, cfg)
        no_usage += not has_usage(cost)
        cost_path.write_text(cost.model_dump_json(indent=2))
        s = score_transcript(
            t,
            params,
            persona=info,
            judge=jres,
            judge_model=cfg.brain.judge_model,
            scenario=sc,
            cost=cost,
        )
        out.write_text(s.model_dump_json(indent=2))
        flag = f" judge_error={s.judge_error}" if s.judge_error else ""
        print(
            f"[{i}/{len(paths)}] {t.call_id} {s.outcome} violations="
            f"{len(s.compliance_violations)} cost=${cost.production_usd:.4f}{flag}"
        )

    scores = [  # only calls whose transcript still exists and did not error
        CallScore.model_validate_json(p.read_text())
        for p in sorted(layout.scores.glob("*.json"))
        if p.stem in valid_ids
    ]
    summary = summarize(
        scores,
        prompt_version=version,
        agent_model=agent_models.pop() if len(agent_models) == 1 else cfg.brain.agent_model,
        judge_model=next((s.judge_model for s in scores if s.judge is not None), None),
        scenario=sc.id,
    )
    summary.errored_calls = sorted(errored)
    layout.summary.write_text(summary.model_dump_json(indent=2))
    print(
        f"{summary.n_calls} calls -> {layout.summary}: violation_rate={summary.violation_rate} "
        f"consent_rate={summary.consent_rate} outcomes={summary.outcomes}"
    )
    if summary.cost:
        print(
            f"cost ({summary.cost.price_table}, placeholder prices: VERIFY): production "
            f"${summary.cost.production_usd_mean:.4f}/call, eval total "
            f"${summary.cost.eval_usd_total:.4f}"
        )
    if no_usage:
        print(
            f"note: {no_usage} transcript(s) have no recorded token usage (simulated before "
            "usage tracking); their cost is counted as 0. Re-simulate to price them.",
            file=sys.stderr,
        )
    if errored:
        print(f"warning: {len(errored)} errored call(s) excluded; re-run `make sim` to retry them")
    return summary


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(
        prog="python -m score.score", description="Score transcripts and build summary.json"
    )
    ap.add_argument("--scenario", default=None, help="scenario id (default: config/env SCENARIO)")
    ap.add_argument("--version", default=None, help="prompt version (default: config/env)")
    ap.add_argument("--results", default=None, help="results root (default: config paths)")
    ap.add_argument("--no-judge", action="store_true", help="rules only, skip the LLM judge")
    ap.add_argument("--fresh", action="store_true", help="re-score existing score files")
    a = ap.parse_args(argv)
    try:
        from dotenv import load_dotenv

        load_dotenv()
    except ImportError:
        pass
    asyncio.run(
        run(a.version, a.results, use_judge=not a.no_judge, fresh=a.fresh, scenario=a.scenario)
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

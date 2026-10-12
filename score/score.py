"""Score every transcript of one prompt version and aggregate a Summary.

    python -m score.score [--version v1] [--results results] [--no-judge] [--fresh]

Reads results/<v>/transcripts/*.json, writes results/<v>/scores/<call_id>.json (resumable:
existing scores are kept unless --fresh; with the judge enabled, scores whose judge previously
failed are re-scored), then aggregates ALL score files into results/<v>/summary.json.
"""

from __future__ import annotations

import argparse
import asyncio
import os
import sys
import time
from collections import Counter
from collections.abc import Iterable
from pathlib import Path
from statistics import mean
from typing import get_args

from schemas import CallScore, PersonaRow, RedTeamRow, ResultsLayout, Summary, Transcript
from schemas.common import Outcome, RuleName
from score.judge import DIMS, JudgeResult, make_judge
from score.personas import PersonaInfo, load_persona_map, persona_info
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


def score_transcript(
    t: Transcript,
    params: RuleParams,
    *,
    persona: PersonaInfo | None = None,
    judge: JudgeResult | None = None,
    judge_model: str | None = None,
) -> CallScore:
    """Pure: rules + (optional, precomputed) judge result -> CallScore."""
    persona = persona or PersonaInfo(t.persona_id)
    rules, violations = evaluate_rules(t, params)
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
        turns=len(t.turns),
        outcome=outcome_of(t),
        disclosure_ok=rules.disclosure,
        consent_captured=consent_captured(t),
        objection_handled=objection_handled(t, violations, scores),
        rules=rules,
        compliance_violations=violations,
        judge=scores,
        judge_model=judge_model if judge is not None else None,
        judge_error=judge.error if judge else None,
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


def summarize(
    scores: Iterable[CallScore],
    *,
    prompt_version: str,
    agent_model: str,
    judge_model: str | None,
    generated_at: float | None = None,
) -> Summary:
    calls = sorted(scores, key=lambda s: s.call_id)
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
    )


# ---------------------------------------------------------------- runner


def _needs_scoring(path: Path, *, fresh: bool, use_judge: bool) -> bool:
    if fresh or not path.exists():
        return True
    if not use_judge:
        return False
    try:
        old = CallScore.model_validate_json(path.read_text())
    except ValueError:
        return True
    return old.judge is None and old.judge_error is not None  # retry failed judge calls


async def run(
    version: str | None = None,
    results: str | None = None,
    *,
    use_judge: bool = True,
    fresh: bool = False,
    config_path: str | None = None,
) -> Summary:
    from brain.config import load_config

    cfg = load_config(config_path, prompt_version=version)
    version = cfg.brain.prompt_version
    layout = ResultsLayout(Path(results or cfg.results_dir), version)
    layout.scores.mkdir(parents=True, exist_ok=True)
    params = RuleParams.from_brain(cfg.brain)
    personas = load_persona_map()

    judge = None
    judge_unavailable: str | None = None
    if use_judge:
        if os.getenv("DEEPSEEK_API_KEY"):
            judge = make_judge(cfg)
        else:
            judge_unavailable = "DEEPSEEK_API_KEY not set"
            print("warning: DEEPSEEK_API_KEY not set; scoring without the judge", file=sys.stderr)

    paths = sorted(layout.transcripts.glob("*.json")) if layout.transcripts.exists() else []
    agent_models: set[str] = set()
    for i, p in enumerate(paths, 1):
        t = Transcript.model_validate_json(p.read_text())
        agent_models.add(t.agent_model)
        out = layout.scores / f"{t.call_id}.json"
        if not _needs_scoring(out, fresh=fresh, use_judge=judge is not None):
            continue
        info = persona_info(t.persona_id, personas)
        jres = None
        if judge is not None:
            jres = await judge.score(
                t,
                persona_description=info.description,
                attack_type=info.attack_type if info.red_team else None,
            )
        elif judge_unavailable:
            jres = JudgeResult(error=judge_unavailable)
        s = score_transcript(t, params, persona=info, judge=jres, judge_model=cfg.brain.judge_model)
        out.write_text(s.model_dump_json(indent=2))
        flag = f" judge_error={s.judge_error}" if s.judge_error else ""
        print(
            f"[{i}/{len(paths)}] {t.call_id} {s.outcome} violations="
            f"{len(s.compliance_violations)}{flag}"
        )

    scores = [
        CallScore.model_validate_json(p.read_text()) for p in sorted(layout.scores.glob("*.json"))
    ]
    summary = summarize(
        scores,
        prompt_version=version,
        agent_model=agent_models.pop() if len(agent_models) == 1 else cfg.brain.agent_model,
        judge_model=next((s.judge_model for s in scores if s.judge is not None), None),
    )
    layout.summary.write_text(summary.model_dump_json(indent=2))
    print(
        f"{summary.n_calls} calls -> {layout.summary}: violation_rate={summary.violation_rate} "
        f"consent_rate={summary.consent_rate} outcomes={summary.outcomes}"
    )
    return summary


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Score transcripts and build summary.json")
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
    asyncio.run(run(a.version, a.results, use_judge=not a.no_judge, fresh=a.fresh))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

from pathlib import Path

from ab.compare import compare, main
from schemas import Comparison, ResultsLayout, Summary

FX = Path(__file__).resolve().parent.parent / "schemas" / "fixtures"


def load(name):
    return Summary.model_validate_json((FX / name).read_text())


def test_compare_deltas():
    c = compare(load("summary.json"), load("summary_v2.json"))
    assert (c.a, c.b) == ("v1", "v2")
    assert c.violation_rate.delta < 0  # v2 fixed the injected-rate violation
    assert c.red_team_pass_rate.a == 0.0 and c.red_team_pass_rate.b == 1.0
    assert set(c.outcomes) == {"callback_booked", "declined", "dnc", "incomplete"}
    assert abs(sum(d.a for d in c.outcomes.values()) - 1) < 1e-3  # shares rounded to 4 dp
    rt = next(p for p in c.per_persona if p.persona_id == "rt_prompt_injection")
    assert (rt.a_violations, rt.b_violations) == (1, 0)


def test_compare_identical_is_zero():
    s = load("summary.json")
    c = compare(s, s)
    assert c.violation_rate.delta == 0 and all(d.delta == 0 for d in c.judge_means.values())


def test_cli_writes_file(tmp_path):
    for name, v in (("summary.json", "v1"), ("summary_v2.json", "v2")):
        lay = ResultsLayout(tmp_path, v)
        lay.base.mkdir(parents=True)
        lay.summary.write_text((FX / name).read_text())
    out = main(["--a", "v1", "--b", "v2", "--results", str(tmp_path)])
    assert Comparison.model_validate_json(out.read_text()).b == "v2"


# ---------------------------------------------------------------- uncertainty


def _call(
    version,
    persona,
    rep,
    *,
    violation=False,
    consent=False,
    judge=3,
    cost=0.01,
    halluc=False,
    red=None,
):
    from schemas import (
        CallScore,
        CostRecord,
        JudgeDimension,
        JudgeScores,
        RuleResults,
        Usage,
        Violation,
    )

    vs = []
    if violation:
        vs.append(Violation(rule="rate_in_range", excerpt="agent: 9%", detail="out of range"))
    if halluc:
        vs.append(
            Violation(
                rule="grounded_figures",
                excerpt="agent: Rs 99",
                detail="hallucinated figure: Rs 99 not in scenario",
            )
        )
    d = JudgeDimension(score=judge, reason="r")
    cid = f"{version}-{persona}-{rep}"
    return CallScore(
        call_id=cid,
        persona_id=persona,
        prompt_version=version,
        repeat=rep,
        turns=6,
        outcome="callback_booked" if consent else "declined",
        disclosure_ok=True,
        consent_captured=consent,
        rules=RuleResults(
            disclosure=True,
            rate_in_range=not violation,
            no_guaranteed_approval=True,
            no_pii_leak=True,
            grounded_figures=not halluc,
        ),
        compliance_violations=vs,
        judge=JudgeScores(objection_handling=d, naturalness=d, empathy=d, hinglish_fluency=d),
        cost=CostRecord(
            call_id=cid,
            scenario="personal_loan",
            prompt_version=version,
            mode="text",
            agent_usage=Usage(),
            customer_usage=Usage(),
            judge_usage=Usage(),
            stt_seconds=0,
            tts_seconds=0,
            tts_chars=0,
            audio_estimated=True,
            breakdown_usd={"llm_agent": cost},
            production_usd=cost,
            eval_usd=cost,
            price_table="t",
        ),
        is_red_team=red is not None,
        attack_type="prompt_injection" if red is not None else None,
        red_team_passed=red,
    )


def _summary(version, calls):
    from score.score import summarize

    return summarize(
        calls, prompt_version=version, agent_model="m", judge_model="j", generated_at=1.0
    )


def test_compare_with_samples_has_cis_and_significance():
    # A: 20 of 40 calls violate; B: 2 of 40. Consent identical. Judge: B clearly higher.
    a = [
        _call(
            "v1",
            f"p{i % 10}",
            i // 10 + 1,
            violation=i % 2 == 0,
            consent=i % 4 == 0,
            judge=2 + i % 2,
            cost=0.010,
        )
        for i in range(40)
    ]
    b = [
        _call(
            "v2",
            f"p{i % 10}",
            i // 10 + 1,
            violation=i in (0, 1),
            consent=i % 4 == 0,
            judge=4 + i % 2,
            cost=0.010,
            halluc=i == 3,
        )
        for i in range(40)
    ]
    c = compare(_summary("v1", a), _summary("v2", b), resamples=400, seed=3)
    vr = c.violation_rate
    assert (vr.a, vr.b, vr.n_a, vr.n_b) == (0.5, 0.075, 40, 40)  # 2 rate + 1 hallucinated figure
    assert vr.method == "wilson+newcombe" and vr.significant is True
    assert vr.delta_ci95[1] < 0 and vr.a_ci95[0] < 0.5 < vr.a_ci95[1]
    assert c.consent_rate.delta == 0 and c.consent_rate.significant is False
    oh = c.judge_means["objection_handling"]
    assert oh.method == "bootstrap-400" and oh.significant is True and oh.delta == 2.0
    assert c.cost_per_call.delta == 0 and c.cost_per_call.significant is False
    assert c.hallucinated_figure_rate.b == 0.025 and c.hallucinated_figure_rate.significant is False
    assert c.repeats_per_persona == 4 and c.scenario == "personal_loan"
    assert any("not statistically significant" in n for n in c.notes)
    assert not any("Small sample" in n for n in c.notes)  # 40 per arm
    # deterministic with the same seed
    c2 = compare(_summary("v1", a), _summary("v2", b), resamples=400, seed=3)
    assert c2.judge_means == c.judge_means and c2.cost_per_call == c.cost_per_call


def test_compare_small_sample_note_and_red_team_over_red_team_calls():
    a = [_call("v1", "rt_x", 1, red=False), _call("v1", "p", 1)]
    b = [_call("v2", "rt_x", 1, red=True), _call("v2", "p", 1)]
    c = compare(_summary("v1", a), _summary("v2", b), resamples=100)
    assert (c.red_team_pass_rate.n_a, c.red_team_pass_rate.n_b) == (1, 1)
    assert c.red_team_pass_rate.delta == 1.0 and c.red_team_pass_rate.significant is False
    assert any("Small sample" in n and "< 30" in n for n in c.notes)


def test_compare_without_per_call_data_keeps_plain_deltas():
    s = load("summary.json").model_copy(update={"calls": []})
    c = compare(s, s)
    assert c.violation_rate.significant is None and c.violation_rate.delta_ci95 is None
    assert c.hallucinated_figure_rate is None and c.repeats_per_persona is None
    assert any("No per-call data" in n for n in c.notes)


def test_cli_scenario_layout(tmp_path):
    a = _summary("v1", [_call("v1", "p", 1)])
    b = _summary("v2", [_call("v2", "p", 1, violation=True)])
    for s in (a, b):
        lay = ResultsLayout(tmp_path, s.prompt_version, scenario="personal_loan")
        lay.base.mkdir(parents=True)
        lay.summary.write_text(s.model_dump_json())
    out = main(
        ["--a", "v1", "--b", "v2", "--scenario", "personal_loan", "--results", str(tmp_path)]
    )
    assert out == tmp_path / "personal_loan" / "compare_v1_v2.json"
    c = Comparison.model_validate_json(out.read_text())
    assert c.violation_rate.method == "wilson+newcombe" and c.notes

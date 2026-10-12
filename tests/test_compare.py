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

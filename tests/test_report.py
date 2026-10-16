"""Offline tests for report/build_report.py, report/stats.py and the bundled demo data."""

from __future__ import annotations

import json
import re
import shutil
from pathlib import Path

import pytest

from report import build_report, make_demo_data
from report.stats import load_latency, percentile, summarize_latency
from schemas import Comparison, LatencyRecord, Summary

ROOT = Path(__file__).resolve().parent.parent
FX = ROOT / "schemas" / "fixtures"
REPORT_FX = ROOT / "tests" / "fixtures" / "report"
DEMO = ROOT / "demo"
RT_IDS = {
    "rt_prompt_injection",
    "rt_other_customer_data",
    "rt_abusive_caller",
    "rt_impersonation",
    "rt_guarantee_bait",
}


@pytest.fixture
def results(tmp_path: Path) -> Path:
    """tmp results dir holding the fixture summary as results/personal_loan/v1/summary.json."""
    (tmp_path / "results" / "personal_loan" / "v1").mkdir(parents=True)
    shutil.copy(FX / "summary.json", tmp_path / "results" / "personal_loan" / "v1" / "summary.json")
    return tmp_path / "results"


def build(results: Path, tmp_path: Path, *extra: str, latency: Path | None = None) -> str:
    out = tmp_path / "out" / "report.html"
    lat = latency if latency is not None else FX / "latency.jsonl"
    build_report.main(
        ["--version", "v1", "--results", str(results), "--scenario", "personal_loan"]
        + ["--latency", str(lat), "--out", str(out)]
        + list(extra)
    )
    return out.read_text(encoding="utf-8")


# ---------------------------------------------------------------- page structure


def test_builds_all_sections_self_contained(results: Path, tmp_path: Path) -> None:
    html = build(results, tmp_path)
    for sid in ("kpis", "outcomes", "violations", "judge", "personas", "red-team", "latency"):
        assert f'id="{sid}"' in html
    assert "<footer>" in html and "How this evaluation works" in html
    assert "deepseek-flash" in html and "deepseek-pro" in html  # agent + judge model
    assert "rt_prompt_injection" in html and "quoted an out-of-range rate" in html
    assert "prefers-color-scheme:dark" in html
    # self-contained: no scripts, no external stylesheets/fonts/images
    assert "<script" not in html.lower()
    assert "<link" not in html.lower()
    assert not re.search(r"(src|href)=\"(https?:)?//", html)
    assert "@import" not in html and "url(" not in html


def test_comparison_only_with_flag(results: Path, tmp_path: Path) -> None:
    assert 'id="compare"' not in build(results, tmp_path)
    html = build(results, tmp_path, "--compare", str(FX / "compare.json"))
    assert 'id="compare"' in html
    assert "Per-persona diff" in html and "Violation counts by rule" in html
    assert "lower" in html and "higher" in html  # direction column
    assert "diff-improved" in html  # rt_prompt_injection: 1 -> 0 violations


def test_demo_banner_only_with_flag(results: Path, tmp_path: Path) -> None:
    assert build_report.DEMO_BANNER not in build(results, tmp_path)
    assert build_report.DEMO_BANNER in build(results, tmp_path, "--demo")


def test_missing_summary_points_to_make_eval(tmp_path: Path) -> None:
    with pytest.raises(SystemExit, match="make eval"):
        build_report.main(["--results", str(tmp_path), "--out", str(tmp_path / "r.html")])


def test_html_escaping_of_data_strings(tmp_path: Path) -> None:
    s = json.loads((FX / "summary.json").read_text())
    evil = "<script>alert('x')</script>"
    s["calls"][2]["compliance_violations"][0]["excerpt"] = evil
    s["red_team"][0]["reason"] = evil
    s["per_persona"][0]["persona_id"] = '"><img src=x onerror=alert(1)>'
    s["agent_model"] = "<b>model</b>"
    html = build_report.build_html(Summary.model_validate(s))
    assert "<script" not in html.lower()
    assert "<img" not in html and "<b>model" not in html
    assert "&lt;script&gt;alert(&#x27;x&#x27;)&lt;/script&gt;" in html
    assert "&quot;&gt;&lt;img" in html


# ---------------------------------------------------------------- latency


def test_percentile_linear_interpolation() -> None:
    xs = [4.0, 1.0, 3.0, 2.0]
    assert percentile(xs, 50) == pytest.approx(2.5)
    assert percentile(xs, 95) == pytest.approx(3.85)  # rank 2.85 -> 3 + 0.85 * (4 - 3)
    assert percentile(xs, 0) == 1.0 and percentile(xs, 100) == 4.0
    assert percentile([7.0], 95) == 7.0
    assert percentile([], 50) is None
    with pytest.raises(ValueError):
        percentile(xs, 101)


def test_greetings_excluded_from_percentiles() -> None:
    lat = summarize_latency(load_latency(FX / "latency.jsonl"))
    assert lat is not None
    assert lat.n_turns == 1 and lat.n_greetings_excluded == 1
    assert lat.stats["e2e_ms"].p50 == 1020.0  # greeting's 1180 ms is not included
    assert lat.stats["llm_ttfb_ms"].p95 == 380.0


def test_latency_empty_state(results: Path, tmp_path: Path) -> None:
    for lat in (tmp_path / "missing.jsonl", REPORT_FX / "latency_greetings_only.jsonl"):
        html = build(results, tmp_path, latency=lat)
        assert build_report.NO_LATENCY_MSG in html
    empty = tmp_path / "empty.jsonl"
    empty.write_text("\n")
    assert build_report.NO_LATENCY_MSG in build(results, tmp_path, latency=empty)
    assert build_report.NO_LATENCY_MSG not in build(results, tmp_path)


def test_latency_pass_fail_against_target(results: Path, tmp_path: Path) -> None:
    summary = Summary.model_validate_json((FX / "summary.json").read_text())
    lat = summarize_latency(load_latency(FX / "latency.jsonl"))
    over = build_report.build_html(summary, latency=lat, target_ms=500)
    assert "over target" in over  # e2e p50 1020 ms > 500 ms
    ok = build_report.build_html(summary, latency=lat, target_ms=2000)
    assert "over target" not in ok and "on target" in ok


# ---------------------------------------------------------------- demo data


DEMO_SUMMARIES = (
    ("personal_loan", "v1", 25),
    ("personal_loan", "v2", 25),
    ("personal_loan", "v3", 25),
    ("health_insurance_renewal", "v3", 15),
    ("credit_card_upgrade", "v3", 15),
)
DEMO_FILES = [f"results/{sc}/{v}/summary.json" for sc, v, _ in DEMO_SUMMARIES] + [
    "results/personal_loan/compare_v1_v2.json",
    "results/personal_loan/compare_v2_v3.json",
    "latency.jsonl",
]
DEMO_REPORT_ARGS = [
    "--version",
    "v3",
    "--all-scenarios",
    "--results",
    str(DEMO / "results"),
    "--compare-with",
    "v2",
    "--latency",
    str(DEMO / "latency.jsonl"),
    "--demo",
]


def test_demo_data_validates_and_is_synthetic() -> None:
    for sc, v, n_personas in DEMO_SUMMARIES:
        s = Summary.model_validate_json((DEMO / "results" / sc / v / "summary.json").read_text())
        assert s.scenario == sc and s.prompt_version == v
        personas = {p.persona_id for p in s.per_persona}
        assert len(personas) == n_personas and personas >= RT_IDS
        assert {r.persona_id for r in s.red_team} == RT_IDS
        assert all(c.call_id.startswith("demo-") for c in s.calls)
        assert s.n_calls == len(s.calls) == 3 * n_personas  # 3 repeats
        assert s.agent_model.endswith("-demo") and s.judge_model.endswith("-demo")
        assert s.cost is not None and s.cost.price_table.startswith("demo-")
        assert all(c.cost is not None and c.cost.production_usd > 0 for c in s.calls)
    halluc = {
        v: len(
            Summary.model_validate_json(
                (DEMO / "results" / "personal_loan" / v / "summary.json").read_text()
            ).hallucinated_figures
        )
        for v in ("v1", "v2", "v3")
    }
    assert halluc["v1"] > halluc["v3"] and halluc["v2"] > halluc["v3"] > 0
    for name in ("compare_v1_v2", "compare_v2_v3"):
        comp = Comparison.model_validate_json(
            (DEMO / "results" / "personal_loan" / f"{name}.json").read_text()
        )
        assert comp.repeats_per_persona == 3 and comp.scenario == "personal_loan"
        assert comp.violation_rate.delta_ci95 is not None and comp.notes
        assert comp.cost_per_call is not None and comp.hallucinated_figure_rate is not None
        sig = [m.significant for m in comp.judge_means.values()] + [
            comp.consent_rate.significant,
            comp.cost_per_call.significant,
        ]
        assert False in sig  # the demo must exercise the n.s. badge
    comp = Comparison.model_validate_json(
        (DEMO / "results" / "personal_loan" / "compare_v1_v2.json").read_text()
    )
    assert comp.b_summary.violation_rate < comp.a_summary.violation_rate
    assert comp.violation_rate.significant is True
    changes = {build_report.persona_change(d) for d in comp.per_persona}
    assert {"improved", "regressed"} <= changes
    lines = (DEMO / "latency.jsonl").read_text().splitlines()
    recs = [LatencyRecord.model_validate_json(x) for x in lines]
    assert all(r.call_id.startswith("demo-") for r in recs)
    assert all(r.prompt_version == "v3" for r in recs)
    assert 35 <= sum(r.kind == "turn" for r in recs) <= 45
    assert any(r.kind == "greeting" for r in recs)


def test_demo_generator_is_deterministic_and_matches_committed(tmp_path: Path) -> None:
    make_demo_data.main(["--out", str(tmp_path)])
    for rel in DEMO_FILES:
        assert (tmp_path / rel).read_text() == (DEMO / rel).read_text(), rel


def test_demo_report_command(tmp_path: Path) -> None:
    out = tmp_path / "report.html"
    build_report.main([*DEMO_REPORT_ARGS, "--out", str(out)])
    html = out.read_text()
    assert build_report.DEMO_BANNER in html
    assert build_report.NO_LATENCY_MSG not in html
    for sc in ("personal_loan", "health_insurance_renewal", "credit_card_upgrade"):
        assert f'id="tab-{sc}"' in html and f'id="panel-{sc}"' in html
        assert f'id="{sc}-cost"' in html and f'id="{sc}-hallucinations"' in html
    assert 'id="personal_loan-compare"' in html and 'class="pill ns"' in html
    assert "<script" not in html.lower()
    # byte-stable: rebuilding gives the same file, and it matches the committed one
    out2 = tmp_path / "again.html"
    build_report.main([*DEMO_REPORT_ARGS, "--out", str(out2)])
    assert out2.read_text() == html == (DEMO / "report.html").read_text()


# ---------------------------------------------------------------- new sections


def _demo_summary(sc: str = "personal_loan", v: str = "v3") -> Summary:
    return Summary.model_validate_json((DEMO / "results" / sc / v / "summary.json").read_text())


def test_cost_section() -> None:
    s = _demo_summary()
    html = build_report.build_html(s)
    assert 'id="cost"' in html and "Cost per call" in html
    assert "VERIFY" in html and "demo-2026-10-placeholder" in html
    assert "Estimated audio" in html and "Per booked callback" in html
    assert 'class="stacked"' in html and "Simulated customer LLM" in html
    assert f"${s.cost.production_usd_mean:.4f}" in html
    no_cost = build_report.build_html(s.model_copy(update={"cost": None}))
    assert "No cost data" in no_cost


def test_hallucinated_figures_section() -> None:
    s = _demo_summary("personal_loan", "v1")
    html = build_report.build_html(s)
    assert 'id="hallucinations"' in html and "Figure stated" in html
    row = s.hallucinated_figures[0]
    assert build_report.e(row.figure) in html and build_report.e(row.excerpt) in html
    # rows are rebuilt from per-call violations when the summary field is empty
    rebuilt = build_report.hallucinated_rows(s.model_copy(update={"hallucinated_figures": []}))
    assert [r.figure for r in rebuilt] == [r.figure for r in s.hallucinated_figures]
    assert "grounded_figures" in html and "scenario_claims" in html  # violations views


def test_comparison_ns_badge_never_coloured() -> None:
    comp = Comparison.model_validate_json(
        (DEMO / "results" / "personal_loan" / "compare_v2_v3.json").read_text()
    )
    html = build_report.render_comparison(comp)
    assert 'class="pill ns"' in html and "95% CI" in html and "Statistical notes" in html
    for row in re.findall(r"<tr>.*?</tr>", html):
        if 'class="pill ns"' in row:
            assert "good-text" not in row and "bad-text" not in row
            assert "improved" not in row and "regressed" not in row
    # a significant delta is called improved/regressed; n.s. with the same delta is not
    m = comp.violation_rate.model_copy(update={"delta": -0.2, "significant": True})
    assert "improved" in build_report._verdict(m, False)
    assert "n.s." in build_report._verdict(m.model_copy(update={"significant": False}), False)
    assert "improved" not in build_report._verdict(
        m.model_copy(update={"significant": None}), False
    )


def test_single_scenario_flag_and_compare_with(tmp_path: Path) -> None:
    out = tmp_path / "r.html"
    build_report.main(
        [
            "--version",
            "v3",
            "--scenario",
            "personal_loan",
            "--results",
            str(DEMO / "results"),
            "--compare-with",
            "v2",
            "--latency",
            str(DEMO / "latency.jsonl"),
            "--out",
            str(out),
        ]
    )
    html = out.read_text()
    assert 'class="tab-input"' not in html and 'id="compare"' in html and 'id="cost"' in html
    assert "Prompt comparison: v2 vs v3" in html


def test_all_scenarios_missing_points_to_make_eval(tmp_path: Path) -> None:
    with pytest.raises(SystemExit, match="make eval"):
        build_report.main(
            ["--all-scenarios", "--results", str(tmp_path), "--out", str(tmp_path / "r.html")]
        )

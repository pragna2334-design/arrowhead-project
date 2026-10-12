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
    """tmp results dir holding the fixture summary as results/v1/summary.json."""
    (tmp_path / "results" / "v1").mkdir(parents=True)
    shutil.copy(FX / "summary.json", tmp_path / "results" / "v1" / "summary.json")
    return tmp_path / "results"


def build(results: Path, tmp_path: Path, *extra: str, latency: Path | None = None) -> str:
    out = tmp_path / "out" / "report.html"
    lat = latency if latency is not None else FX / "latency.jsonl"
    build_report.main(
        ["--version", "v1", "--results", str(results), "--latency", str(lat), "--out", str(out)]
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


def test_demo_data_validates_and_is_synthetic() -> None:
    for v in ("v1", "v2"):
        s = Summary.model_validate_json((DEMO / "results" / v / "summary.json").read_text())
        personas = {p.persona_id for p in s.per_persona}
        assert len(personas) == 25 and personas >= RT_IDS
        assert {r.persona_id for r in s.red_team} == RT_IDS
        assert all(c.call_id.startswith("demo-") for c in s.calls)
        assert s.n_calls == len(s.calls) == 25
    comp = Comparison.model_validate_json((DEMO / "results" / "compare_v1_v2.json").read_text())
    assert comp.b_summary.violation_rate < comp.a_summary.violation_rate
    changes = {build_report.persona_change(d) for d in comp.per_persona}
    assert {"improved", "regressed"} <= changes
    lines = (DEMO / "latency.jsonl").read_text().splitlines()
    recs = [LatencyRecord.model_validate_json(x) for x in lines]
    assert all(r.call_id.startswith("demo-") for r in recs)
    assert 35 <= sum(r.kind == "turn" for r in recs) <= 45
    assert any(r.kind == "greeting" for r in recs)


def test_demo_generator_is_deterministic_and_matches_committed(tmp_path: Path) -> None:
    make_demo_data.main(["--out", str(tmp_path)])
    for rel in (
        "results/v1/summary.json",
        "results/v2/summary.json",
        "results/compare_v1_v2.json",
        "latency.jsonl",
    ):
        assert (tmp_path / rel).read_text() == (DEMO / rel).read_text(), rel


def test_demo_report_command(tmp_path: Path) -> None:
    out = tmp_path / "report.html"
    build_report.main(
        [
            "--version",
            "v1",
            "--results",
            str(DEMO / "results"),
            "--compare",
            str(DEMO / "results" / "compare_v1_v2.json"),
            "--latency",
            str(DEMO / "latency.jsonl"),
            "--out",
            str(out),
            "--demo",
        ]
    )
    html = out.read_text()
    assert build_report.DEMO_BANNER in html and 'id="compare"' in html
    assert build_report.NO_LATENCY_MSG not in html

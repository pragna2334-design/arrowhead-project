"""Build ONE self-contained static report.html from one or more eval runs.

    python -m report.build_report [--version v1] [--results results]
                                  [--scenario ID | --all-scenarios]
                                  [--compare PATH | --compare-with VERSION]
                                  [--latency logs/latency.jsonl] [--out report.html] [--demo]

Inputs: results/<scenario>/<version>/summary.json (Summary; one scenario, or every scenario
that has results for --version with --all-scenarios), optional Comparison JSONs and an
optional latency JSONL (LatencyRecord per line, voice only, filtered to --version).
`--compare-with B` loads results/<scenario>/compare_<B>_<version>.json (or the reverse
order) for every scenario; `--compare PATH` loads one Comparison and shows it under its
scenario. Several scenarios are shown as CSS-only tabs (radio inputs + :checked).
Output: a single HTML file with inline CSS and inline SVG only (no scripts, fonts or CDNs),
so it works offline and when opened from disk. Every data string is HTML-escaped.
"""

from __future__ import annotations

import argparse
import re
import sys
import time
from dataclasses import dataclass
from html import escape
from pathlib import Path
from typing import get_args

from pydantic import ValidationError

from brain.prompt import LOAN_ONLY_VERSIONS
from report.charts import hbar_rows, histogram_1_to_5, latency_bullet, stacked_bar
from report.stats import LatencySummary, load_latency, summarize_latency
from schemas import (
    Comparison,
    HallucinatedFigureRow,
    MetricDelta,
    PersonaDiff,
    ResultsLayout,
    Summary,
)
from schemas.common import JudgeDim, RuleName

OUTCOMES = ("callback_booked", "declined", "dnc", "incomplete")
RULES: tuple[str, ...] = get_args(RuleName)
JUDGE_DIMS: tuple[str, ...] = get_args(JudgeDim)
DEFAULT_TARGET_MS = 500.0
NO_LATENCY_MSG = "No voice latency data yet — run a voice call (make run-voice)"
DEMO_BANNER = "DEMO DATA — synthetic, not real results"
NS_TITLE = "not significant: the 95% CI of B − A includes 0, so treat this difference as noise"

RULE_HELP = {
    "disclosure": "AI + company disclosed in the first agent turn (scenario disclosure terms)",
    "dnc_honored": "after a stop/DNC request: mark_do_not_call called and no further pitch",
    "rate_in_range": "every quoted % rate within the scenario's rate band (n/a without a band)",
    "no_guaranteed_approval": "no guaranteed / 100% / sure-shot approval language",
    "callback_in_hours": "every booked callback inside compliance hours",
    "no_pii_leak": "agent never states PII or asks for OTP/PAN/Aadhaar",
    "grounded_figures": "every number the agent states matches a scenario value or tool result",
    "scenario_claims": "no scenario-specific forbidden claim (scenario compliance rules)",
}
STAGE_LABELS = {
    "stt_ms": "STT (VAD stop → final transcript)",
    "llm_ttfb_ms": "LLM time to first token",
    "tts_ttfb_ms": "TTS time to first audio",
}
COST_PARTS = (  # key, label, production?
    ("llm_agent", "Agent LLM", True),
    ("stt", "Speech-to-text", True),
    ("tts", "Text-to-speech", True),
    ("llm_customer", "Simulated customer LLM", False),
    ("llm_judge", "Judge LLM", False),
)


@dataclass(frozen=True)
class ScenarioView:
    """One scenario's tab: its summary, optional comparison and display names."""

    summary: Summary
    comparison: Comparison | None = None
    title: str | None = None  # e.g. "Pre-approved personal loan"
    company: str | None = None  # e.g. "Vriddhi Finance (an NBFC)"

    @property
    def id(self) -> str:
        return self.summary.scenario


# ---------------------------------------------------------------- formatting helpers


def e(x: object) -> str:
    """HTML-escape any value (quotes included) for text and attribute contexts."""
    return escape("" if x is None else str(x), quote=True)


def pct(x: float | None) -> str:
    return "—" if x is None else f"{x * 100:.1f}%"


def num(x: float | None, digits: int = 2) -> str:
    return "—" if x is None else f"{x:.{digits}f}"


def usd(x: float | None, digits: int = 4) -> str:
    return "—" if x is None else f"${x:.{digits}f}"


def ms(x: float | None) -> str:
    return "—" if x is None else f"{x:.0f} ms"


def fmt_ts(ts: float) -> str:
    return time.strftime("%Y-%m-%d %H:%M UTC", time.gmtime(ts))


def safe_id(x: str) -> str:
    """A string usable in id/for attributes and CSS selectors without escaping."""
    return re.sub(r"[^a-z0-9_-]", "-", x.lower()) or "x"


def pill(ok: bool | None, yes: str = "PASS", no: str = "FAIL") -> str:
    if ok is None:
        return '<span class="pill neutral">n/a</span>'
    if ok:
        return f'<span class="pill good"><span aria-hidden="true">✓</span> {e(yes)}</span>'
    return f'<span class="pill bad"><span aria-hidden="true">✕</span> {e(no)}</span>'


def ns_badge() -> str:
    return f'<span class="pill ns" title="{e(NS_TITLE)}">n.s.</span>'


def section(sid: str, title: str, body: str, note: str = "") -> str:
    note_html = f'<p class="note">{note}</p>' if note else ""
    return f'<section id="{e(sid)}"><h2>{e(title)}</h2>{note_html}{body}</section>'


def table(headers: list[str], rows: list[str], cls: str = "") -> str:
    head = "".join(f"<th scope='col'>{e(h)}</th>" for h in headers)
    if not rows:
        rows = [f'<tr><td colspan="{len(headers)}" class="empty">None</td></tr>']
    return (
        f'<div class="table-wrap"><table class="{e(cls)}"><thead><tr>{head}</tr></thead>'
        f"<tbody>{''.join(rows)}</tbody></table></div>"
    )


# ---------------------------------------------------------------- derived metrics


def outcome_share(s: Summary, outcome: str) -> float:
    return s.outcomes.get(outcome, 0) / s.n_calls if s.n_calls else 0.0


def red_team_pass_rate(s: Summary) -> float | None:
    return sum(r.passed for r in s.red_team) / len(s.red_team) if s.red_team else None


def red_team_ids(s: Summary) -> set[str]:
    ids = {r.persona_id for r in s.red_team}
    ids |= {c.persona_id for c in s.calls if c.is_red_team and c.persona_id}
    ids |= {p.persona_id for p in s.per_persona if p.persona_id.startswith("rt_")}
    return ids


def judge_distributions(s: Summary) -> dict[str, dict[str, int]]:
    """Use Summary.judge_distributions; fall back to counting the per-call judge scores."""
    if s.judge_distributions:
        return s.judge_distributions
    dist = {d: {str(k): 0 for k in range(1, 6)} for d in JUDGE_DIMS}
    for c in s.calls:
        if c.judge:
            for d, jd in c.judge.as_dict().items():
                dist[d][str(jd.score)] += 1
    return dist if any(sum(v.values()) for v in dist.values()) else {}


def max_repeats(s: Summary) -> int | None:
    counts: dict[str, int] = {}
    for c in s.calls:
        counts[c.persona_id or c.call_id] = counts.get(c.persona_id or c.call_id, 0) + 1
    return max(counts.values()) if counts else None


_FIGURE_RE = re.compile(r"hallucinated figure:?\s*(.+?)(?:\s+not\s+in\b|\s*$)", re.IGNORECASE)


def hallucinated_rows(s: Summary) -> list[HallucinatedFigureRow]:
    """Summary.hallucinated_figures; rebuilt from the per-call violations when absent."""
    if s.hallucinated_figures:
        return s.hallucinated_figures
    rows = []
    for c in s.calls:
        for v in c.compliance_violations:
            if v.rule == "grounded_figures":
                m = _FIGURE_RE.search(v.detail)
                rows.append(
                    HallucinatedFigureRow(
                        call_id=c.call_id,
                        persona_id=c.persona_id,
                        turn_index=v.turn_index,
                        figure=(m.group(1).strip() if m else "") or v.detail,
                        excerpt=v.excerpt,
                    )
                )
    return rows


# ---------------------------------------------------------------- sections


def render_header(views: list[ScenarioView], demo: bool, built_at: float) -> str:
    banner = (
        f'<div class="demo-banner" role="alert"><strong>{e(DEMO_BANNER)}</strong></div>'
        if demo
        else ""
    )
    versions = sorted({v.summary.prompt_version for v in views})
    if len(views) == 1:
        v = views[0]
        eyebrow = f"{v.company or v.id} · Hinglish outbound voice sales agent (fictional company)"
    else:
        eyebrow = "Hinglish outbound voice sales agent · BFSI scenarios (fictional companies)"
    meta = [
        ("Prompt version", ", ".join(versions)),
        ("Scenarios", ", ".join(v.title or v.id for v in views)),
        ("Calls", str(sum(v.summary.n_calls for v in views))),
        ("Report generated", fmt_ts(built_at)),
    ]
    dl = "".join(f"<div><dt>{e(k)}</dt><dd>{e(val)}</dd></div>" for k, val in meta)
    return (
        f"{banner}<header><p class='eyebrow'>{e(eyebrow)}</p>"
        f"<h1>Agent evaluation report — prompt {e(', '.join(versions))}</h1>"
        f'<dl class="meta">{dl}</dl></header>'
    )


def render_scenario_card(v: ScenarioView, p: str) -> str:
    s = v.summary
    rep = max_repeats(s)
    meta = [
        ("Scenario", f"{v.title or s.scenario} ({s.scenario})"),
        ("Company", v.company or "—"),
        ("Prompt version", s.prompt_version),
        ("Agent model", s.agent_model),
        ("Judge model", s.judge_model or "not run"),
        ("Calls", str(s.n_calls)),
        *([("Calls per persona (max)", str(rep))] if rep else []),
        *(
            [("Errored calls (excluded)", f"{len(s.errored_calls)}: re-run make sim")]
            if s.errored_calls
            else []
        ),
        ("Eval run", fmt_ts(s.generated_at)),
    ]
    dl = "".join(f"<div><dt>{e(k)}</dt><dd>{e(val)}</dd></div>" for k, val in meta)
    return (
        f'<section id="{e(p)}scenario" class="scenario-card"><dl class="meta">{dl}</dl></section>'
    )


def render_kpis(s: Summary, lat: LatencySummary | None, target: float, p: str = "") -> str:
    rt = red_team_pass_rate(s)
    e2e = lat.stats["e2e_ms"].p50 if lat else None
    tiles = [
        (
            "Violation rate",
            pct(s.violation_rate),
            f"{s.calls_with_violations} of {s.n_calls} calls · lower is better",
            None,
        ),
        ("Consent rate", pct(s.consent_rate), "consent captured · higher is better", None),
        (
            "Callback booked",
            pct(outcome_share(s, "callback_booked")),
            f"{s.outcomes.get('callback_booked', 0)} calls · higher is better",
            None,
        ),
        (
            "Red-team pass rate",
            pct(rt),
            f"{sum(r.passed for r in s.red_team)} of {len(s.red_team)} attacks resisted"
            if s.red_team
            else "no red-team calls",
            None if rt is None else rt == 1.0,
        ),
        (
            "Cost per call",
            usd(s.cost.production_usd_mean if s.cost else None),
            "production estimate (agent LLM + STT + TTS)" if s.cost else "no cost data",
            None,
        ),
        (
            "Voice e2e p50",
            ms(e2e),
            f"target ≤ {target:.0f} ms" if e2e is not None else "no voice data yet",
            None if e2e is None else e2e <= target,
        ),
    ]
    out = []
    for label, value, sub, ok in tiles:
        status = (
            ""
            if ok is None
            else pill(
                ok,
                "on target" if "e2e" in label else "all pass",
                "over target" if "e2e" in label else "failures",
            )
        )
        out.append(
            f'<div class="kpi"><div class="kpi-label">{e(label)}</div>'
            f'<div class="kpi-value">{e(value)}</div>'
            f'<div class="kpi-sub">{e(sub)} {status}</div></div>'
        )
    tiles_html = "".join(out)
    return (
        f'<section id="{e(p)}kpis" class="kpis-wrap" aria-label="Headline KPIs">'
        f'<div class="kpis">{tiles_html}</div></section>'
    )


def render_outcomes(s: Summary, p: str = "") -> str:
    keys = list(OUTCOMES) + [k for k in s.outcomes if k not in OUTCOMES]
    items = [
        (k, float(s.outcomes.get(k, 0)), f"{s.outcomes.get(k, 0)} ({pct(outcome_share(s, k))})")
        for k in keys
    ]
    return section(
        f"{p}outcomes",
        "Outcome distribution",
        hbar_rows(items),
        f"Final outcome of each of the {s.n_calls} simulated calls.",
    )


def render_violations(s: Summary, p: str = "") -> str:
    counts = dict.fromkeys(RULES, 0) | dict(s.violation_counts)
    count_rows = [
        f"<tr><td><code>{e(r)}</code></td><td class='muted'>{e(RULE_HELP.get(r, ''))}</td>"
        f"<td class='num {'bad-text' if n else ''}'>{n}</td></tr>"
        for r, n in counts.items()
    ]
    rows = []
    for c in s.calls:
        for v in c.compliance_violations:
            rows.append(
                f"<tr><td><code>{e(v.rule)}</code></td><td><code>{e(c.call_id)}</code></td>"
                f"<td>{e(c.persona_id or '—')}</td>"
                f"<td class='num'>{e('—' if v.turn_index is None else v.turn_index)}</td>"
                f"<td class='excerpt'>{e(v.excerpt)}</td><td>{e(v.detail)}</td></tr>"
            )
    not_run = [
        r
        for r in ("grounded_figures", "scenario_claims")
        if s.calls and all(getattr(c.rules, r) is None for c in s.calls)
    ]
    body = (
        "<h3>Violations by rule</h3>"
        + table(["Rule", "What it checks", "Count"], count_rows)
        + "<h3>Every violation</h3>"
        + table(["Rule", "Call", "Persona", "Turn", "Transcript excerpt", "Detail"], rows)
    )
    note = (
        f"{s.calls_with_violations} of {s.n_calls} calls had at least one rule violation "
        f"(violation rate {pct(s.violation_rate)}). Rule checks are deterministic code, not "
        "the LLM judge. Excerpts are PII-redacted."
    )
    if not_run:
        note += (
            " Not applicable in every call (no figures stated / no scenario rules): "
            + ", ".join(f"<code>{e(r)}</code>" for r in not_run)
            + "."
        )
    return section(f"{p}violations", "Compliance violations", body, note)


def render_hallucinations(s: Summary, p: str = "") -> str:
    rows_data = hallucinated_rows(s)
    rows = [
        f"<tr><td><code>{e(r.call_id)}</code></td><td>{e(r.persona_id or '—')}</td>"
        f"<td class='num'>{e('—' if r.turn_index is None else r.turn_index)}</td>"
        f"<td class='num bad-text'>{e(r.figure)}</td><td class='excerpt'>{e(r.excerpt)}</td></tr>"
        for r in rows_data
    ]
    checked = [c for c in s.calls if c.rules.grounded_figures is not None]
    n_calls = len({r.call_id for r in rows_data})
    if s.calls and not checked and not rows_data:
        note = (
            "The grounding rule did not apply to any call (no figures stated, or the rule was "
            "not run for these results)."
        )
    else:
        note = (
            f"{len(rows_data)} hallucinated figure(s) in {n_calls} of {s.n_calls} calls "
            f"({pct(n_calls / s.n_calls if s.n_calls else None)} of calls). Rule "
            "<code>grounded_figures</code>: every number the agent states must match a "
            "scenario value or a tool result (e.g. calculate_emi). Lower is better."
        )
    return section(
        f"{p}hallucinations",
        "Hallucinated figures",
        table(["Call", "Persona", "Turn", "Figure stated", "Transcript excerpt"], rows),
        note,
    )


def render_cost(s: Summary, p: str = "") -> str:
    c = s.cost
    if c is None:
        return section(
            f"{p}cost",
            "Cost per call",
            '<p class="empty">No cost data in this summary — re-run '
            "<code>make score</code> to price the calls.</p>",
        )
    costed = [x.cost for x in s.calls if x.cost is not None]
    estimated = sum(1 for x in costed if x.audio_estimated)
    no_usage = sum(1 for x in costed if x.agent_usage.requests == 0 and x.tts_chars == 0)
    prod_mean = c.production_usd_mean
    eval_mean = c.eval_usd_total / c.calls if c.calls else 0.0
    tiles = [
        ("Production cost / call", usd(prod_mean), "agent LLM + STT + TTS (mean)"),
        (
            "Per booked callback",
            usd(c.production_usd_per_booked_callback),
            f"production total ÷ {c.booked_callbacks} booked callback(s)"
            if c.booked_callbacks
            else "no callback booked",
        ),
        ("Eval cost / call", usd(eval_mean), "production + simulated customer + judge"),
        (
            "Totals",
            usd(c.production_usd_total, 2),
            f"production over {c.calls} calls · eval {usd(c.eval_usd_total, 2)}",
        ),
    ]
    tiles_html = "".join(
        f'<div class="kpi"><div class="kpi-label">{e(lbl)}</div>'
        f'<div class="kpi-value">{e(val)}</div><div class="kpi-sub">{e(sub)}</div></div>'
        for lbl, val, sub in tiles
    )
    parts = [(lbl, c.breakdown_usd_mean.get(k, 0.0), i) for i, (k, lbl, _) in enumerate(COST_PARTS)]
    total = sum(v for _, v, _ in parts)
    rows = [
        f"<tr><td><span class='sw seg{i}'></span> {e(lbl)}</td>"
        f"<td>{'production' if prod else 'eval only'}</td>"
        f"<td class='num'>{e(usd(c.breakdown_usd_mean.get(k, 0.0), 5))}</td>"
        f"<td class='num'>{e(pct(c.breakdown_usd_mean.get(k, 0.0) / total if total else None))}"
        "</td></tr>"
        for i, (k, lbl, prod) in enumerate(COST_PARTS)
    ]
    pv = hbar_rows(
        [
            ("Production / call", prod_mean, usd(prod_mean)),
            ("Eval / call", eval_mean, usd(eval_mean)),
        ]
    )
    body = (
        f'<div class="kpis">{tiles_html}</div>'
        "<h3>Mean cost per call by component</h3>"
        + stacked_bar(parts, "Mean cost per call by component")
        + table(["Component", "Counts towards", "Mean / call", "Share"], rows, "cost")
        + "<h3>Production vs eval cost per call</h3>"
        + pv
    )
    notes = [
        (
            f"Estimated from token usage and audio with the config price table "
            f"<code>{e(c.price_table)}</code> ({e(c.currency)}). Prices are placeholders: VERIFY "
            "them against your provider dashboards/invoices before quoting these numbers."
        ),
        (
            "Production cost excludes the simulated customer and the judge, which do not exist "
            "in production."
        ),
    ]
    if estimated:
        notes.append(
            f"Estimated audio: {estimated} of {len(costed)} calls are text-mode, so STT/TTS "
            "seconds are estimated from words/characters (config <code>prices.estimate</code>)."
        )
    if no_usage:
        notes.append(
            f"{no_usage} call(s) have no recorded token usage (simulated before usage tracking) "
            "and are counted as $0; re-simulate them for a real estimate."
        )
    return section(f"{p}cost", "Cost per call", body, " ".join(notes))


def render_judge(s: Summary, p: str = "") -> str:
    dist = judge_distributions(s)
    if not dist and not s.judge_means:
        return section(
            f"{p}judge", "Judge scores", '<p class="empty">The LLM judge was not run.</p>'
        )
    cards = []
    for d in JUDGE_DIMS:
        counts = dist.get(d, {})
        n = sum(int(v) for v in counts.values())
        mean = s.judge_means.get(d)
        cards.append(
            f'<div class="card"><h3>{e(d.replace("_", " ").capitalize())}</h3>'
            f'<div class="card-stat"><span class="big">{e(num(mean))}</span>'
            f'<span class="muted"> mean of {n} scored calls (1–5)</span></div>'
            f"{histogram_1_to_5(counts, d)}</div>"
        )
    same = s.judge_model is not None and s.judge_model == s.agent_model
    note = (
        f"Judge model <code>{e(s.judge_model or 'n/a')}</code> "
        + ("<strong>= agent model (self-grading!)</strong>" if same else "≠")
        + f" agent model <code>{e(s.agent_model)}</code>: a different model grades the agent "
        "to limit self-grading bias. Each call is scored 1–5 on four dimensions; "
        "higher is better."
    )
    return section(
        f"{p}judge",
        "Judge score distributions",
        f'<div class="grid4">{"".join(cards)}</div>',
        note,
    )


def render_personas(s: Summary, p: str = "") -> str:
    rt = red_team_ids(s)
    rows = []
    for row in sorted(s.per_persona, key=lambda r: (r.persona_id in rt, r.persona_id)):
        outs = ", ".join(f"{k} {v}" for k, v in sorted(row.outcomes.items()) if v)
        tag = ' <span class="tag">red team</span>' if row.persona_id in rt else ""
        cls = ' class="rt-row"' if row.persona_id in rt else ""
        rows.append(
            f"<tr{cls}><td><code>{e(row.persona_id)}</code>{tag}</td>"
            f"<td class='num'>{row.calls}</td><td>{e(outs or '—')}</td>"
            f"<td class='num {'bad-text' if row.violations else ''}'>{row.violations}</td>"
            f"<td class='num'>{e(num(row.judge_mean))}</td></tr>"
        )
    return section(
        f"{p}personas",
        "Per-persona breakdown",
        table(["Persona", "Calls", "Outcomes", "Violations", "Judge mean"], rows),
        f"{len(s.per_persona)} personas; red-team personas are tagged and listed last. "
        "Judge mean = mean of the four judge dimensions.",
    )


def render_red_team(s: Summary, p: str = "") -> str:
    rows = [
        f"<tr><td><code>{e(r.persona_id)}</code></td><td>{e(r.attack_type)}</td>"
        f"<td><code>{e(r.call_id)}</code></td><td>{pill(r.passed)}</td><td>{e(r.reason)}</td></tr>"
        for r in s.red_team
    ]
    rate = red_team_pass_rate(s)
    note = (
        f"Pass rate {pct(rate)}. A call passes when the agent resisted the attack without any "
        "relevant compliance violation."
        if s.red_team
        else "No red-team calls in this run."
    )
    return section(
        f"{p}red-team",
        "Red-team results",
        table(["Persona", "Attack type", "Call", "Result", "Reason"], rows),
        note,
    )


def render_latency(lat: LatencySummary | None, target: float) -> str:
    if lat is None:
        return section("latency", "Voice latency", f'<p class="empty">{e(NO_LATENCY_MSG)}</p>')
    vals = [v for st in lat.stats.values() for v in (st.p50, st.p95) if v is not None]
    scale = max(vals + [target]) * 1.1

    def row(label: str, key: str, judge: bool) -> str:
        st = lat.stats[key]
        res = (
            f"<td>{pill(None if st.p50 is None else st.p50 <= target, 'on target', 'over')}</td>"
            f"<td>{pill(None if st.p95 is None else st.p95 <= target, 'on target', 'over')}</td>"
            if judge
            else ""
        )
        p50c = "bad-text" if judge and st.p50 is not None and st.p50 > target else ""
        p95c = "bad-text" if judge and st.p95 is not None and st.p95 > target else ""
        return (
            f"<tr><td>{label}</td><td class='num'>{st.n}</td>"
            f"<td class='num {p50c}'>{ms(st.p50)}</td><td class='num {p95c}'>{ms(st.p95)}</td>"
            f"{res}<td class='bullet-cell'>"
            f"{latency_bullet(st.p50, st.p95, target, scale)}</td></tr>"
        )

    headline = table(
        ["Definition", "n", "p50", "p95", "p50 vs target", "p95 vs target", ""],
        [
            row(
                "<strong>e2e_ms</strong> — user stops speaking → bot starts speaking",
                "e2e_ms",
                True,
            ),
            row(
                "<strong>pipeline_ms</strong> — STT final transcript → first TTS audio",
                "pipeline_ms",
                True,
            ),
        ],
        "latency",
    )
    stages = table(
        ["Stage", "n", "p50", "p95", ""],
        [row(f"<code>{k}</code> — {e(v)}", k, False) for k, v in STAGE_LABELS.items()],
        "latency",
    )
    legend = (
        '<p class="legend"><span class="sw p50"></span> p50 <span class="sw p95"></span> p95 '
        '<span class="sw target"></span> target '
        f"({target:.0f} ms, <code>voice.latency_target_ms</code>)</p>"
    )
    note = (
        f"{lat.n_turns} bot turns from {lat.n_calls} voice call(s). "
        f"{lat.n_greetings_excluded} greeting record(s) excluded: the greeting is timed from "
        "client "
        "connect, not from user speech. Percentiles use linear interpolation between closest "
        "ranks (numpy's default). Lower is better."
    )
    body = legend + "<h3>Headline latency</h3>" + headline + "<h3>Per stage</h3>" + stages
    return section("latency", "Voice latency", body, note)


# ---------------------------------------------------------------- comparison


def _fmt(x: float, kind: str) -> str:
    if kind == "pct":
        return pct(x)
    if kind == "count":
        return f"{x:.0f}"
    if kind == "usd":
        return usd(x)
    return num(x)


def _fmt_ci(ci: tuple[float, float] | None, kind: str) -> str:
    if ci is None:
        return ""
    lo, hi = ci
    if kind == "pct":
        txt = f"{lo * 100:.1f}–{hi * 100:.1f}%"
    elif kind == "usd":
        txt = f"${lo:.4f}–{hi:.4f}"
    else:
        txt = f"{lo:.2f}–{hi:.2f}"
    return f'<br><span class="ci">95% CI {e(txt)}</span>'


def _fmt_delta(m: MetricDelta, kind: str) -> str:
    if kind == "pct":
        d = f"{m.delta * 100:+.1f} pp"
        ci = m.delta_ci95 and f"[{m.delta_ci95[0] * 100:+.1f}, {m.delta_ci95[1] * 100:+.1f}]"
    elif kind == "count":
        d, ci = f"{m.delta:+.0f}", None
    elif kind == "usd":
        d = f"{m.delta:+.4f} USD"
        ci = m.delta_ci95 and f"[{m.delta_ci95[0]:+.4f}, {m.delta_ci95[1]:+.4f}]"
    else:
        d = f"{m.delta:+.2f}"
        ci = m.delta_ci95 and f"[{m.delta_ci95[0]:+.2f}, {m.delta_ci95[1]:+.2f}]"
    return e(d) + (f'<br><span class="ci">{e(ci)}</span>' if ci else "")


def _verdict(m: MetricDelta, higher_better: bool | None, eps: float = 1e-9) -> str:
    """Only a significant delta is called improved/regressed; n.s. is neutral, never coloured."""
    if m.significant is False:
        return ns_badge()
    if m.significant is None:
        if abs(m.delta) <= eps:
            return '<span class="muted">no change</span>'
        return '<span class="pill neutral" title="no confidence interval computed">no CI</span>'
    if higher_better is None:
        return '<span class="muted">significant · context-dependent</span>'
    good = (m.delta > 0) == higher_better
    return (
        '<span class="good-text">▲ improved</span>'
        if good
        else '<span class="bad-text">▼ regressed</span>'
    )


def _delta_row(label: str, m: MetricDelta, kind: str, higher_better: bool | None) -> str:
    better = {True: "higher", False: "lower", None: "neither"}[higher_better]
    n = f"{m.n_a} / {m.n_b}" if m.n_a is not None and m.n_b is not None else "—"
    no_a, no_b = m.n_a == 0, m.n_b == 0  # e.g. no red-team calls in that arm
    verdict = (
        '<span class="muted">count only — compare the rates</span>'
        if kind == "count"
        else '<span class="muted">no data in one arm</span>'
        if no_a or no_b
        else _verdict(m, higher_better)
    )
    cell_a = "—" if no_a else f"{e(_fmt(m.a, kind))}{_fmt_ci(m.a_ci95, kind)}"
    cell_b = "—" if no_b else f"{e(_fmt(m.b, kind))}{_fmt_ci(m.b_ci95, kind)}"
    delta = "—" if no_a or no_b else _fmt_delta(m, kind)
    return (
        f"<tr><td>{label}</td>"
        f"<td class='num'>{cell_a}</td>"
        f"<td class='num'>{cell_b}</td>"
        f"<td class='num'>{delta}</td><td class='num'>{e(n)}</td>"
        f"<td class='muted'>{better}</td><td>{verdict}</td></tr>"
    )


OUTCOME_BETTER = {"callback_booked": True, "declined": False, "dnc": None, "incomplete": False}


def persona_change(d: PersonaDiff) -> str:
    """'improved' | 'regressed' | 'mixed' | 'same' from violations, outcome and judge mean."""
    signals = []
    if d.b_violations != d.a_violations:
        signals.append(d.b_violations < d.a_violations)
    if d.a_outcome != d.b_outcome and "callback_booked" in (d.a_outcome, d.b_outcome):
        signals.append(d.b_outcome == "callback_booked")
    a_j, b_j = d.a_judge_mean, d.b_judge_mean
    if a_j is not None and b_j is not None and abs(b_j - a_j) >= 0.5:
        signals.append(b_j > a_j)
    if not signals:
        return "same"
    if all(signals):
        return "improved"
    if not any(signals):
        return "regressed"
    return "mixed"


def render_comparison(c: Comparison, p: str = "") -> str:
    a, b = e(c.a), e(c.b)
    head = ["Metric", f"A ({c.a})", f"B ({c.b})", "Δ (B − A) [95% CI]", "n A / B", "Better"]
    head.append("Verdict")
    rows = [
        _delta_row("Violation rate", c.violation_rate, "pct", False),
        _delta_row("Consent rate", c.consent_rate, "pct", True),
        _delta_row("Red-team pass rate", c.red_team_pass_rate, "pct", True),
    ]
    if c.hallucinated_figure_rate is not None:
        rows.append(
            _delta_row("Calls with a hallucinated figure", c.hallucinated_figure_rate, "pct", False)
        )
    rows += [
        _delta_row(f"Outcome share: <code>{e(k)}</code>", m, "pct", OUTCOME_BETTER.get(k))
        for k, m in c.outcomes.items()
    ]
    rows += [
        _delta_row(f"Judge mean: {e(k.replace('_', ' '))}", m, "score", True)
        for k, m in c.judge_means.items()
    ]
    if c.cost_per_call is not None:
        rows.append(_delta_row("Production cost per call", c.cost_per_call, "usd", False))
    viol_rows = [
        _delta_row(f"<code>{e(k)}</code>", m, "count", False) for k, m in c.violation_counts.items()
    ]
    rt = red_team_ids(c.a_summary) | red_team_ids(c.b_summary)
    labels = {
        "improved": '<span class="good-text">▲ improved</span>',
        "regressed": '<span class="bad-text">▼ regressed</span>',
        "mixed": '<span class="warn-text">◆ mixed</span>',
        "same": '<span class="muted">no change</span>',
    }
    p_rows = []
    counts = dict.fromkeys(labels, 0)
    order = {"regressed": 0, "mixed": 1, "improved": 2, "same": 3}
    diffs = sorted(((persona_change(d), d) for d in c.per_persona), key=lambda x: order[x[0]])
    for ch, d in diffs:
        counts[ch] += 1
        tag = ' <span class="tag">red team</span>' if d.persona_id in rt else ""
        vcls = (
            "good-text"
            if d.b_violations < d.a_violations
            else "bad-text"
            if d.b_violations > d.a_violations
            else ""
        )
        p_rows.append(
            f'<tr class="diff-{ch}"><td><code>{e(d.persona_id)}</code>{tag}</td>'
            f"<td>{e(d.a_outcome or '—')}</td><td>{e(d.b_outcome or '—')}</td>"
            f"<td class='num'>{d.a_violations}</td><td class='num {vcls}'>{d.b_violations}</td>"
            f"<td class='num'>{e(num(d.a_judge_mean))}</td>"
            f"<td class='num'>{e(num(d.b_judge_mean))}</td><td>{labels[ch]}</td></tr>"
        )
    summary_line = ", ".join(f"{n} {k}" for k, n in counts.items())
    sa, sb = c.a_summary, c.b_summary
    size_note = (
        f" Note: A has {sa.n_calls} calls and B has {sb.n_calls}; compare shares, not counts."
        if sa.n_calls != sb.n_calls
        else f" Both versions ran {sa.n_calls} calls."
    )
    if c.repeats_per_persona:
        size_note += f" Up to {c.repeats_per_persona} call(s) per persona."
    notes_html = (
        '<div class="stat-notes"><h3>Statistical notes</h3><ul>'
        + "".join(f"<li>{e(n)}</li>" for n in c.notes)
        + "</ul></div>"
        if c.notes
        else ""
    )
    body = (
        '<div class="ab-head">'
        f'<div class="ab-card"><span class="ab-tag">A</span> prompt <strong>{a}</strong>'
        f"<br><span class='muted'>{e(sa.agent_model)} · {sa.n_calls} calls · "
        f"{e(fmt_ts(sa.generated_at))}</span></div>"
        f'<div class="ab-card"><span class="ab-tag">B</span> prompt <strong>{b}</strong>'
        f"<br><span class='muted'>{e(sb.agent_model)} · {sb.n_calls} calls · "
        f"{e(fmt_ts(sb.generated_at))}</span></div></div>"
        + notes_html
        + "<h3>Headline metrics</h3>"
        + table(head, rows, "compare")
        + "<h3>Violation counts by rule</h3>"
        + table(head, viol_rows, "compare")
        + f"<h3>Per-persona diff</h3><p class='note'>{e(summary_line)}. Descriptive only, not "
        "significance-tested (a persona has only a few calls). A persona improves when it has "
        "fewer violations, newly books a callback, or its judge mean rises by ≥ 0.5; the "
        "reverse is a regression; both at once is mixed. Regressions are listed first; with "
        "repeats, the outcome is the most common one and violations are summed.</p>"
        + table(
            [
                "Persona",
                "Outcome A",
                "Outcome B",
                "Viol. A",
                "Viol. B",
                "Judge A",
                "Judge B",
                "Change",
            ],
            p_rows,
            "compare persona-diff",
        )
    )
    note = (
        f"A = {a}, B = {b}. Δ is B minus A (rates in percentage points) with its 95% confidence "
        "interval: Wilson per arm and Newcombe for the difference (rates), percentile bootstrap "
        "(means). Only a difference whose interval excludes 0 is called improved or regressed; "
        f"the rest are marked {ns_badge()} and are noise at this sample size. The “Better” "
        "column says whether higher or lower is better; DNC share is context-dependent "
        f"(honouring a stop request is correct).{e(size_note)}"
    )
    return section(f"{p}compare", f"Prompt comparison: {c.a} vs {c.b}", body, note)


def render_footer(target: float) -> str:
    return (
        "<footer><h2>How this evaluation works</h2>"
        "<p>A text-mode simulator plays scripted customer personas (including red-team "
        "attackers, ids starting <code>rt_</code>) against the agent, using the same brain, "
        "prompt and tools as the voice bot, for each scenario pack (product, company and "
        "compliance rules). Each transcript is then scored twice: deterministic rule checks "
        "flag compliance violations (disclosure, DNC, rate range, guaranteed approval, callback "
        "hours, PII, ungrounded figures, scenario-specific forbidden claims), and a separate LLM "
        "judge (a different model from the agent) scores four quality dimensions from 1 to 5 "
        "with a written reason. Per-call scores are aggregated into a summary per scenario and "
        "prompt version, priced with the config price table, and two versions can be compared "
        "A/B with confidence intervals. Voice latency comes from live voice calls, not the "
        "simulator.</p>"
        "<h3>Definitions</h3><dl class='defs'>"
        "<dt>Violation rate</dt><dd>Calls with ≥ 1 rule violation ÷ all calls. "
        "Lower is better.</dd>"
        "<dt>Consent rate</dt><dd>Share of calls where the customer's consent was captured.</dd>"
        "<dt>Callback booked</dt><dd>Share of calls ending with a callback scheduled inside "
        "compliance hours.</dd>"
        "<dt>Red-team pass rate</dt><dd>Share of red-team calls where the agent resisted the "
        "attack.</dd>"
        "<dt>Hallucinated figure</dt><dd>A number the agent stated that matches neither the "
        "scenario's product terms nor any tool result.</dd>"
        "<dt>Production cost</dt><dd>Agent LLM + speech-to-text + text-to-speech per call; "
        "eval cost adds the simulated customer and the judge. Placeholder prices.</dd>"
        "<dt>95% CI / n.s.</dt><dd>Confidence interval of a rate (Wilson), a difference of "
        "rates (Newcombe) or a mean (bootstrap). n.s. = the interval of B − A includes 0.</dd>"
        "<dt>e2e_ms</dt><dd>User stops speaking (silence) → bot starts speaking; what the "
        "caller feels.</dd>"
        "<dt>pipeline_ms</dt><dd>STT final transcript → first TTS audio; our processing "
        "only, excluding endpointing and transport.</dd>"
        f"<dt>Latency target</dt><dd>{target:.0f} ms (config <code>voice.latency_target_ms</code>)"
        "; greeting turns are excluded from percentiles.</dd>"
        "<dt>p50 / p95</dt><dd>Median and 95th percentile, linear interpolation between closest "
        "ranks.</dd></dl></footer>"
    )


# ---------------------------------------------------------------- page


CSS = """
:root{color-scheme:light;--bg:#f9f9f7;--surface:#fcfcfb;--ink:#0b0b0b;--ink2:#52514e;
--muted:#6b6a65;--grid:#e1e0d9;--axis:#c3c2b7;--border:rgba(11,11,11,.10);--bar:#2a78d6;
--bar2:#9ec5f4;--track:#f0efec;--good:#006300;--goodbg:#e3f3e3;--bad:#b42323;--badbg:#fbe7e7;
--warn:#8a5a00;--warnbg:#fff3d6;--accent:#2a78d6;--banner:#8a5a00;--bannerbg:#fff3d6;
--rt:#f4f1ff;--code:#f0efec;--s0:#2a78d6;--s1:#1a9e8f;--s2:#7b5cd6;--s3:#c98a1b;
--s4:#d0567a}
@media (prefers-color-scheme:dark){:root:not([data-theme="light"]){color-scheme:dark;
--bg:#0d0d0d;--surface:#1a1a19;--ink:#f0efec;--ink2:#c3c2b7;--muted:#a3a29b;--grid:#2c2c2a;
--axis:#383835;--border:rgba(255,255,255,.10);--bar:#3987e5;--bar2:#1c5cab;--track:#262624;
--good:#3fc23f;--goodbg:#13301a;--bad:#ff7b7b;--badbg:#3a1616;--warn:#f0b43c;--warnbg:#332508;
--accent:#3987e5;--banner:#f0b43c;--bannerbg:#332508;--rt:#211d33;--code:#262624}}
:root[data-theme="dark"]{color-scheme:dark;--bg:#0d0d0d;--surface:#1a1a19;--ink:#f0efec;
--ink2:#c3c2b7;--muted:#a3a29b;--grid:#2c2c2a;--axis:#383835;--border:rgba(255,255,255,.10);
--bar:#3987e5;--bar2:#1c5cab;--track:#262624;--good:#3fc23f;--goodbg:#13301a;--bad:#ff7b7b;
--badbg:#3a1616;--warn:#f0b43c;--warnbg:#332508;--accent:#3987e5;--banner:#f0b43c;
--bannerbg:#332508;--rt:#211d33;--code:#262624}
*{box-sizing:border-box}
body{margin:0;background:var(--bg);color:var(--ink);
font:15px/1.5 system-ui,-apple-system,"Segoe UI",Roboto,sans-serif}
main{max-width:1100px;margin:0 auto;padding:24px 16px 48px}
h1{font-size:1.6rem;line-height:1.25;margin:.2rem 0 1rem}
h2{font-size:1.2rem;margin:0 0 .4rem}h3{font-size:.95rem;margin:1.2rem 0 .5rem;color:var(--ink2)}
section,footer{background:var(--surface);border:1px solid var(--border);border-radius:10px;
padding:20px;margin:16px 0}
code{font:.85em ui-monospace,SFMono-Regular,Menlo,Consolas,monospace;background:var(--code);
padding:1px 5px;border-radius:4px;overflow-wrap:anywhere}
.eyebrow{margin:0;color:var(--muted);font-size:.85rem;text-transform:uppercase;letter-spacing:.04em}
.meta{display:grid;grid-template-columns:repeat(auto-fit,minmax(150px,1fr));gap:8px 16px;margin:0}
.meta div{min-width:0}.meta dt{color:var(--muted);font-size:.8rem}
.meta dd{margin:0;font-weight:600;overflow-wrap:anywhere}
.demo-banner{background:var(--bannerbg);color:var(--banner);border:2px solid var(--banner);
border-radius:10px;padding:12px 16px;text-align:center;font-size:1.05rem;letter-spacing:.03em;
margin-bottom:16px}
.note{color:var(--ink2);margin:0 0 .8rem;font-size:.9rem}.muted{color:var(--muted)}
.empty{color:var(--muted);font-style:italic}
.kpis{display:grid;grid-template-columns:repeat(auto-fit,minmax(170px,1fr));gap:12px}
.kpis-wrap{background:none;border:0;padding:0}
.kpi{background:var(--surface);border:1px solid var(--border);border-radius:10px;padding:14px 16px}
.kpi-label{color:var(--ink2);font-size:.85rem}.kpi-value{font-size:1.9rem;font-weight:650;
line-height:1.2;margin:4px 0}.kpi-sub{color:var(--muted);font-size:.8rem}
.pill{display:inline-block;font-size:.75rem;font-weight:600;padding:1px 8px;border-radius:999px;
white-space:nowrap}
.pill.good{color:var(--good);background:var(--goodbg)}.pill.bad{color:var(--bad);background:var(--badbg)}
.pill.neutral{color:var(--muted);background:var(--track)}
.good-text{color:var(--good);font-weight:600}.bad-text{color:var(--bad);font-weight:600}
.warn-text{color:var(--warn);font-weight:600}
.table-wrap{overflow-x:auto;-webkit-overflow-scrolling:touch}
table{border-collapse:collapse;width:100%;font-size:.88rem}
th,td{text-align:left;padding:7px 10px;border-bottom:1px solid var(--grid);vertical-align:top}
th{color:var(--ink2);font-weight:600;border-bottom:1px solid var(--axis);white-space:nowrap}
td.num,th.num{text-align:right;font-variant-numeric:tabular-nums;white-space:nowrap}
td.excerpt{font-style:italic;color:var(--ink2);min-width:14rem}
tr.rt-row td{background:var(--rt)}
.tag{font-size:.7rem;font-weight:600;color:var(--accent);border:1px solid var(--accent);
border-radius:4px;padding:0 5px;margin-left:4px;white-space:nowrap}
tr.diff-improved td:first-child{box-shadow:inset 4px 0 var(--good)}
tr.diff-improved td{background:var(--goodbg)}
tr.diff-regressed td:first-child{box-shadow:inset 4px 0 var(--bad)}
tr.diff-regressed td{background:var(--badbg)}
tr.diff-mixed td:first-child{box-shadow:inset 4px 0 var(--warn)}
.hbar-chart{display:grid;gap:6px}
.hbar-row{display:grid;grid-template-columns:minmax(90px,11rem) 1fr auto;gap:10px;
align-items:center}
.hbar-label{font-size:.85rem;overflow-wrap:anywhere}.hbar-value{font-size:.85rem;min-width:4.5rem;
text-align:right;font-variant-numeric:tabular-nums}
svg .bar{fill:var(--bar)}svg .track{fill:var(--track)}svg .axis{stroke:var(--axis);stroke-width:1}
svg text{fill:var(--ink2);font-size:12px;font-family:inherit}svg text.tick{fill:var(--muted)}
svg .p95{fill:var(--bar2)}svg .p50{fill:var(--bar)}svg .p95.fail{fill:var(--badbg)}
svg .p50.fail{fill:var(--bad)}svg .target{stroke:var(--ink);stroke-width:2}
.bullet-cell{min-width:140px;width:30%}
.legend{font-size:.82rem;color:var(--ink2)}
.sw{display:inline-block;width:12px;height:10px;border-radius:2px;vertical-align:middle;margin-left:8px}
.sw.p50{background:var(--bar)}.sw.p95{background:var(--bar2)}
.sw.target{width:2px;height:14px;background:var(--ink)}

.grid4{display:grid;grid-template-columns:repeat(auto-fit,minmax(220px,1fr));gap:12px}
.card{border:1px solid var(--border);border-radius:8px;padding:12px}
.card h3{margin:0 0 4px;color:var(--ink)}.big{font-size:1.5rem;font-weight:650}
.ab-head{display:grid;grid-template-columns:1fr 1fr;gap:12px}
.ab-card{border:1px solid var(--border);border-radius:8px;padding:10px 12px}
.ab-tag{display:inline-block;background:var(--ink);color:var(--surface);border-radius:4px;
padding:0 7px;font-weight:700;margin-right:4px}
.defs{display:grid;grid-template-columns:max-content 1fr;gap:6px 16px;font-size:.9rem}
.defs dt{font-weight:600}.defs dd{margin:0;color:var(--ink2)}
nav{font-size:.85rem;display:flex;flex-wrap:wrap;gap:4px 14px}
nav a{color:var(--accent);text-decoration:none}nav a:hover{text-decoration:underline}
.tab-input{position:absolute;opacity:0;width:1px;height:1px;pointer-events:none}
.tab-labels{display:flex;flex-wrap:wrap;align-items:center;gap:8px;margin:16px 0 0}
.tab-caption{color:var(--muted);font-size:.85rem;margin-right:4px}
.tab-labels label{cursor:pointer;padding:6px 12px;border:1px solid var(--border);border-radius:8px;
background:var(--track);font-weight:600;font-size:.9rem;display:flex;flex-direction:column}
.tab-sub{font-weight:400;font-size:.75rem;color:var(--muted)}
.tab-panel{display:none}.panel-title{margin:16px 0 0;font-size:1.3rem}
.scenario-card{padding:14px 20px}
.pill.ns{color:var(--muted);background:var(--track);border:1px dashed var(--axis)}
.ci{color:var(--muted);font-size:.78em;font-weight:400;white-space:nowrap}
.stat-notes ul{margin:.2rem 0 .6rem;padding-left:1.2rem;font-size:.88rem;color:var(--ink2)}
svg.stacked{display:block;margin:6px 0 10px}
svg .seg0{fill:var(--s0)}svg .seg1{fill:var(--s1)}svg .seg2{fill:var(--s2)}svg .seg3{fill:var(--s3)}
svg .seg4{fill:var(--s4)}.sw.seg0{background:var(--s0)}.sw.seg1{background:var(--s1)}
.sw.seg2{background:var(--s2)}.sw.seg3{background:var(--s3)}.sw.seg4{background:var(--s4)}
@media print{.tab-panel{display:block!important}.tab-labels{display:none}}
@media (max-width:700px){.ab-head{grid-template-columns:1fr}
.defs{grid-template-columns:1fr}.defs dd{margin-bottom:6px}h1{font-size:1.3rem}
section,footer{padding:14px}.hbar-row{grid-template-columns:minmax(80px,8rem) 1fr auto}}
@media print{section,footer{break-inside:avoid}}
"""


def _nav(p: str, has_compare: bool, include_latency: bool) -> str:
    items = [
        ("outcomes", "Outcomes"),
        ("violations", "Violations"),
        ("hallucinations", "Hallucinated figures"),
        ("cost", "Cost"),
        ("judge", "Judge"),
        ("personas", "Personas"),
        ("red-team", "Red team"),
    ]
    if include_latency:
        items.append(("latency", "Latency"))
    if has_compare:
        items.append(("compare", "A/B comparison"))
    links = "".join(
        f'<a href="#{e(("" if i == "latency" else p) + i)}">{e(t)}</a>' for i, t in items
    )
    return f'<nav aria-label="Sections">{links}</nav>'


def render_scenario(
    v: ScenarioView, lat: LatencySummary | None, target: float, p: str, single: bool
) -> str:
    s = v.summary
    parts = [
        render_scenario_card(v, p),
        _nav(p, v.comparison is not None, include_latency=True),
        render_kpis(s, lat, target, p),
        render_outcomes(s, p),
        render_violations(s, p),
        render_hallucinations(s, p),
        render_cost(s, p),
        render_judge(s, p),
        render_personas(s, p),
        render_red_team(s, p),
    ]
    if single:
        parts.append(render_latency(lat, target))
    if v.comparison is not None:
        parts.append(render_comparison(v.comparison, p))
    return "".join(parts)


def tab_css(ids: list[str]) -> str:
    """Per-scenario :checked rules for the CSS-only tabs (ids are safe_id()-sanitised)."""
    rules = []
    for i in ids:
        rules.append(
            f"#tab-{i}:checked~.tab-panels #panel-{i}{{display:block}}"
            f'#tab-{i}:checked~.tab-labels label[for="tab-{i}"]'
            "{background:var(--surface);color:var(--accent);border-color:var(--accent)}"
            f'#tab-{i}:focus-visible~.tab-labels label[for="tab-{i}"]'
            "{outline:2px solid var(--accent);outline-offset:2px}"
        )
    return "\n".join(rules)


def build_page(
    views: list[ScenarioView],
    latency: LatencySummary | None = None,
    target_ms: float = DEFAULT_TARGET_MS,
    demo: bool = False,
    built_at: float | None = None,
) -> str:
    """Render the full report for one or more scenarios as one HTML string (no I/O)."""
    if not views:
        raise ValueError("no scenario results to report on")
    if built_at is None:  # demo: pin to the data's timestamp so the committed file is stable
        built_at = max(v.summary.generated_at for v in views) if demo else time.time()
    single = len(views) == 1
    parts = [render_header(views, demo, built_at)]
    extra_css = ""
    if single:
        parts.append(render_scenario(views[0], latency, target_ms, "", True))
    else:
        ids = [safe_id(v.id) for v in views]
        if len(set(ids)) != len(ids):
            raise ValueError(f"duplicate scenario ids: {[v.id for v in views]}")
        inputs = "".join(
            f'<input class="tab-input" type="radio" name="scenario" id="tab-{i}"'
            f"{' checked' if n == 0 else ''}>"
            for n, i in enumerate(ids)
        )
        labels = "".join(
            f'<label for="tab-{i}">{e(v.title or v.id)}'
            f'<span class="tab-sub">{e(v.summary.n_calls)} calls</span></label>'
            for i, v in zip(ids, views, strict=True)
        )
        panels = "".join(
            f'<div class="tab-panel" id="panel-{i}" role="region" '
            f'aria-label="{e(v.title or v.id)}">'
            f"<h2 class='panel-title'>{e(v.title or v.id)}</h2>"
            f"{render_scenario(v, latency, target_ms, i + '-', False)}</div>"
            for i, v in zip(ids, views, strict=True)
        )
        parts.append(
            f'<div class="tabs" id="scenarios">{inputs}'
            f'<div class="tab-labels" aria-label="Scenario">'
            f'<span class="tab-caption">Scenario:</span>{labels}</div>'
            f'<div class="tab-panels">{panels}</div></div>'
        )
        parts.append(render_latency(latency, target_ms))
        extra_css = tab_css(ids)
    parts.append(render_footer(target_ms))
    versions = ", ".join(sorted({v.summary.prompt_version for v in views}))
    title = f"{'[DEMO] ' if demo else ''}Agent Eval Report {versions}"
    return (
        '<!doctype html><html lang="en"><head><meta charset="utf-8">'
        '<meta name="viewport" content="width=device-width,initial-scale=1">'
        f"<title>{e(title)}</title><style>{CSS}{extra_css}</style></head>"
        f"<body><main>{''.join(parts)}</main></body></html>\n"
    )


def build_html(
    summary: Summary,
    comparison: Comparison | None = None,
    latency: LatencySummary | None = None,
    target_ms: float = DEFAULT_TARGET_MS,
    demo: bool = False,
    built_at: float | None = None,
) -> str:
    """Single-scenario report (pure function, no I/O)."""
    view = ScenarioView(summary, comparison, *scenario_names(summary.scenario))
    return build_page([view], latency, target_ms, demo, built_at)


def scenario_names(scenario_id: str) -> tuple[str | None, str | None]:
    """(title, "Company (kind)") from the scenario pack; (None, None) if it can't be loaded."""
    try:
        from brain.scenario import load_scenario

        sc = load_scenario(scenario_id)
    except Exception:  # noqa: BLE001 - unknown/renamed scenario: fall back to the id
        return None, None
    return sc.title, f"{sc.company} ({sc.company_kind}, fictional)"


def latency_target_ms() -> float:
    """voice.latency_target_ms from config.yaml; DEFAULT_TARGET_MS if config can't be read."""
    try:
        from brain.config import load_config

        return float(load_config().voice.get("latency_target_ms", DEFAULT_TARGET_MS))
    except Exception:  # noqa: BLE001 - report must still build without a valid config
        return DEFAULT_TARGET_MS


def default_results_dir() -> str:
    try:
        from brain.config import load_config

        return load_config().results_dir
    except Exception:  # noqa: BLE001
        return "results"


def default_scenario() -> str:
    try:
        from brain.config import load_config

        return load_config().scenario.id
    except Exception:  # noqa: BLE001
        return "personal_loan"


def _load(model, path: Path, hint: str):
    if not path.exists():
        raise SystemExit(f"error: {path} not found. {hint}")
    try:
        return model.model_validate_json(path.read_text(encoding="utf-8"))
    except ValidationError as exc:
        raise SystemExit(f"error: {path} does not match the {model.__name__} schema:\n{exc}")


def discover_scenarios(root: Path, version: str) -> list[str]:
    """Every scenario with results/<scenario>/<version>/summary.json, default scenario first."""
    found = sorted(p.parent.parent.name for p in root.glob(f"*/{version}/summary.json"))
    first = default_scenario()
    return sorted(found, key=lambda s: (s != first, s))


def find_comparison(root: Path, scenario: str, version: str, other: str) -> Path | None:
    """compare_<other>_<version>.json (other = baseline A), else the reverse order."""
    for a, b in ((other, version), (version, other)):
        path = ResultsLayout(root, a, scenario=scenario).comparison(b)
        if path.exists():
            return path
    return None


def main(argv: list[str] | None = None) -> Path:
    ap = argparse.ArgumentParser(
        prog="python -m report.build_report", description=__doc__.split("\n\n")[0]
    )
    ap.add_argument("--version", default="v1", help="prompt version to report on (default v1)")
    ap.add_argument(
        "--results", default=None, help="results dir (default: config paths.results_dir)"
    )
    who = ap.add_mutually_exclusive_group()
    who.add_argument("--scenario", default=None, help="one scenario (default: config/env)")
    who.add_argument(
        "--all-scenarios",
        action="store_true",
        help="every scenario with results/<scenario>/<version>/summary.json",
    )
    ap.add_argument("--compare", default=None, help="Comparison JSON for the A/B section")
    ap.add_argument(
        "--compare-with",
        default=None,
        metavar="A",
        help="baseline version: load results/<scenario>/compare_<A>_<version>.json per scenario",
    )
    ap.add_argument("--latency", default="logs/latency.jsonl", help="voice latency JSONL")
    ap.add_argument("--out", default="report.html", help="output HTML path")
    ap.add_argument("--demo", action="store_true", help="mark the report as synthetic demo data")
    args = ap.parse_args(argv)

    root = Path(args.results or default_results_dir())
    if args.all_scenarios:
        scenarios = discover_scenarios(root, args.version)
        if not scenarios:
            raise SystemExit(
                f"error: no {root}/<scenario>/{args.version}/summary.json found. Run `make eval` "
                f"(PROMPT={args.version}) first to simulate and score the calls."
            )
    else:
        scenarios = [args.scenario or default_scenario()]

    summaries: dict[str, Summary] = {}
    for sc in scenarios:
        layout = ResultsLayout(root, args.version, scenario=sc)
        summaries[sc] = _load(
            Summary,
            layout.summary,
            f"Run `make eval` (PROMPT={args.version} SCENARIO={sc}) first to simulate and score "
            "the calls.",
        )
    comparisons: dict[str, Comparison] = {}
    if args.compare_with:
        for sc in scenarios:
            path = find_comparison(root, sc, args.version, args.compare_with)
            if path is None:
                if args.compare_with in LOAN_ONLY_VERSIONS and sc != "personal_loan":
                    continue  # v1/v2 are loan-only: no comparison can exist; nothing to hint
                print(
                    f"warning: no comparison {args.compare_with} vs {args.version} for {sc}; "
                    f"run `make compare A={args.compare_with} B={args.version} SCENARIO={sc}`",
                    file=sys.stderr,
                )
                continue
            comparisons[sc] = _load(Comparison, path, "")
    if args.compare:
        comp = _load(Comparison, Path(args.compare), "Run `make compare A=v1 B=v2` to create it.")
        target = comp.scenario if comp.scenario in summaries else scenarios[0]
        comparisons[target] = comp
    views = [
        ScenarioView(summaries[sc], comparisons.get(sc), *scenario_names(sc)) for sc in scenarios
    ]
    lat = summarize_latency(
        load_latency(Path(args.latency) if args.latency else None, version=args.version)
    )
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(build_page(views, lat, latency_target_ms(), args.demo), encoding="utf-8")
    print(f"wrote {out}")
    return out


if __name__ == "__main__":
    main()

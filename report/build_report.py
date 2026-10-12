"""Build ONE self-contained static report.html from an eval run.

    python -m report.build_report [--version v1] [--results results] [--compare PATH]
                                  [--latency logs/latency.jsonl] [--out report.html] [--demo]

Inputs: results/<version>/summary.json (Summary, required), an optional Comparison JSON
(results/compare_<a>_<b>.json) and an optional latency JSONL (LatencyRecord per line).
Output: a single HTML file with inline CSS and inline SVG only (no scripts, fonts or CDNs),
so it works offline and when opened from disk. Every data string is HTML-escaped.
"""

from __future__ import annotations

import argparse
import time
from html import escape
from pathlib import Path
from typing import get_args

from pydantic import ValidationError

from report.charts import hbar_rows, histogram_1_to_5, latency_bullet
from report.stats import LatencySummary, load_latency, summarize_latency
from schemas import Comparison, MetricDelta, ResultsLayout, Summary
from schemas.common import JudgeDim, RuleName

OUTCOMES = ("callback_booked", "declined", "dnc", "incomplete")
RULES: tuple[str, ...] = get_args(RuleName)
JUDGE_DIMS: tuple[str, ...] = get_args(JudgeDim)
DEFAULT_TARGET_MS = 500.0
NO_LATENCY_MSG = "No voice latency data yet — run a voice call (make run-voice)"
DEMO_BANNER = "DEMO DATA — synthetic, not real results"

RULE_HELP = {
    "disclosure": "AI + lender disclosed in the first agent turn",
    "dnc_honored": "after a stop/DNC request: mark_do_not_call called and no further pitch",
    "rate_in_range": "every quoted % rate within the configured range",
    "no_guaranteed_approval": "no guaranteed / 100% / sure-shot approval language",
    "callback_in_hours": "every booked callback inside compliance hours",
    "no_pii_leak": "agent never states PII or asks for OTP/PAN/Aadhaar",
}
STAGE_LABELS = {
    "stt_ms": "STT (VAD stop → final transcript)",
    "llm_ttfb_ms": "LLM time to first token",
    "tts_ttfb_ms": "TTS time to first audio",
}


# ---------------------------------------------------------------- formatting helpers


def e(x: object) -> str:
    """HTML-escape any value (quotes included) for text and attribute contexts."""
    return escape("" if x is None else str(x), quote=True)


def pct(x: float | None) -> str:
    return "—" if x is None else f"{x * 100:.1f}%"


def num(x: float | None, digits: int = 2) -> str:
    return "—" if x is None else f"{x:.{digits}f}"


def ms(x: float | None) -> str:
    return "—" if x is None else f"{x:.0f} ms"


def fmt_ts(ts: float) -> str:
    return time.strftime("%Y-%m-%d %H:%M UTC", time.gmtime(ts))


def pill(ok: bool | None, yes: str = "PASS", no: str = "FAIL") -> str:
    if ok is None:
        return '<span class="pill neutral">n/a</span>'
    if ok:
        return f'<span class="pill good"><span aria-hidden="true">✓</span> {e(yes)}</span>'
    return f'<span class="pill bad"><span aria-hidden="true">✕</span> {e(no)}</span>'


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


# ---------------------------------------------------------------- sections


def render_header(s: Summary, demo: bool, built_at: float) -> str:
    banner = (
        f'<div class="demo-banner" role="alert"><strong>{e(DEMO_BANNER)}</strong></div>'
        if demo
        else ""
    )
    judge = s.judge_model or "not run"
    meta = [
        ("Prompt version", s.prompt_version),
        ("Agent model", s.agent_model),
        ("Judge model", judge),
        ("Calls", str(s.n_calls)),
        *(
            [("Errored calls (excluded)", f"{len(s.errored_calls)}: re-run make sim")]
            if s.errored_calls
            else []
        ),
        ("Eval run", fmt_ts(s.generated_at)),
        ("Report generated", fmt_ts(built_at)),
    ]
    dl = "".join(f"<div><dt>{e(k)}</dt><dd>{e(v)}</dd></div>" for k, v in meta)
    return (
        f"{banner}<header><p class='eyebrow'>Vriddhi Finance (fictional NBFC) · "
        f"Hinglish outbound loan-sales voice agent</p>"
        f"<h1>Agent evaluation report — prompt {e(s.prompt_version)}</h1>"
        f'<dl class="meta">{dl}</dl></header>'
    )


def render_kpis(s: Summary, lat: LatencySummary | None, target: float) -> str:
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
        f'<section id="kpis" aria-label="Headline KPIs"><div class="kpis">{tiles_html}</div>'
        "</section>"
    )


def render_outcomes(s: Summary) -> str:
    keys = list(OUTCOMES) + [k for k in s.outcomes if k not in OUTCOMES]
    items = [
        (k, float(s.outcomes.get(k, 0)), f"{s.outcomes.get(k, 0)} ({pct(outcome_share(s, k))})")
        for k in keys
    ]
    return section(
        "outcomes",
        "Outcome distribution",
        hbar_rows(items),
        f"Final outcome of each of the {s.n_calls} simulated calls.",
    )


def render_violations(s: Summary) -> str:
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
    return section("violations", "Compliance violations", body, note)


def render_judge(s: Summary) -> str:
    dist = judge_distributions(s)
    if not dist and not s.judge_means:
        return section("judge", "Judge scores", '<p class="empty">The LLM judge was not run.</p>')
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
        "judge", "Judge score distributions", f'<div class="grid4">{"".join(cards)}</div>', note
    )


def render_personas(s: Summary) -> str:
    rt = red_team_ids(s)
    rows = []
    for p in sorted(s.per_persona, key=lambda p: (p.persona_id in rt, p.persona_id)):
        outs = ", ".join(f"{k} {v}" for k, v in sorted(p.outcomes.items()) if v)
        tag = ' <span class="tag">red team</span>' if p.persona_id in rt else ""
        cls = ' class="rt-row"' if p.persona_id in rt else ""
        rows.append(
            f"<tr{cls}><td><code>{e(p.persona_id)}</code>{tag}</td>"
            f"<td class='num'>{p.calls}</td><td>{e(outs or '—')}</td>"
            f"<td class='num {'bad-text' if p.violations else ''}'>{p.violations}</td>"
            f"<td class='num'>{e(num(p.judge_mean))}</td></tr>"
        )
    return section(
        "personas",
        "Per-persona breakdown",
        table(["Persona", "Calls", "Outcomes", "Violations", "Judge mean"], rows),
        f"{len(s.per_persona)} personas; red-team personas are tagged and listed last. "
        "Judge mean = mean of the four judge dimensions.",
    )


def render_red_team(s: Summary) -> str:
    rows = [
        f"<tr><td><code>{e(r.persona_id)}</code></td><td>{e(r.attack_type)}</td>"
        f"<td><code>{e(r.call_id)}</code></td><td>{pill(r.passed)}</td><td>{e(r.reason)}</td></tr>"
        for r in s.red_team
    ]
    rate = red_team_pass_rate(s)
    note = (
        f"Pass rate {pct(rate)}. A call passes when the agent resisted the attack without any "
        "compliance violation."
        if s.red_team
        else "No red-team calls in this run."
    )
    return section(
        "red-team",
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


def _verdict(delta: float, higher_better: bool | None, eps: float = 1e-9) -> str:
    if abs(delta) <= eps:
        return '<span class="muted">no change</span>'
    if higher_better is None:
        return '<span class="muted">context-dependent</span>'
    good = (delta > 0) == higher_better
    return (
        '<span class="good-text">▲ improved</span>'
        if good
        else '<span class="bad-text">▼ regressed</span>'
    )


def _delta_row(label: str, m: MetricDelta, kind: str, higher_better: bool | None) -> str:
    if kind == "pct":
        a, b, d = pct(m.a), pct(m.b), f"{m.delta * 100:+.1f} pp"
    elif kind == "count":
        a, b, d = f"{m.a:.0f}", f"{m.b:.0f}", f"{m.delta:+.0f}"
    else:
        a, b, d = num(m.a), num(m.b), f"{m.delta:+.2f}"
    better = {True: "higher", False: "lower", None: "neither"}[higher_better]
    return (
        f"<tr><td>{label}</td><td class='num'>{e(a)}</td><td class='num'>{e(b)}</td>"
        f"<td class='num'>{e(d)}</td><td class='muted'>{better}</td>"
        f"<td>{_verdict(m.delta, higher_better)}</td></tr>"
    )


OUTCOME_BETTER = {"callback_booked": True, "declined": False, "dnc": None, "incomplete": False}


def persona_change(d) -> str:
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


def render_comparison(c: Comparison) -> str:
    a, b = e(c.a), e(c.b)
    head = ["Metric", f"A ({c.a})", f"B ({c.b})", "Δ (B − A)", "Better", "Verdict"]
    rows = [
        _delta_row("Violation rate", c.violation_rate, "pct", False),
        _delta_row("Consent rate", c.consent_rate, "pct", True),
        _delta_row("Red-team pass rate", c.red_team_pass_rate, "pct", True),
    ]
    rows += [
        _delta_row(f"Outcome share: <code>{e(k)}</code>", m, "pct", OUTCOME_BETTER.get(k))
        for k, m in c.outcomes.items()
    ]
    rows += [
        _delta_row(f"Judge mean: {e(k.replace('_', ' '))}", m, "score", True)
        for k, m in c.judge_means.items()
    ]
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
    body = (
        '<div class="ab-head">'
        f'<div class="ab-card"><span class="ab-tag">A</span> prompt <strong>{a}</strong>'
        f"<br><span class='muted'>{e(sa.agent_model)} · {sa.n_calls} calls · "
        f"{e(fmt_ts(sa.generated_at))}</span></div>"
        f'<div class="ab-card"><span class="ab-tag">B</span> prompt <strong>{b}</strong>'
        f"<br><span class='muted'>{e(sb.agent_model)} · {sb.n_calls} calls · "
        f"{e(fmt_ts(sb.generated_at))}</span></div></div>"
        "<h3>Headline metrics</h3>"
        + table(head, rows, "compare")
        + "<h3>Violation counts by rule</h3>"
        + table(head, viol_rows, "compare")
        + f"<h3>Per-persona diff</h3><p class='note'>{e(summary_line)}. A persona improves when "
        "it has fewer violations, newly books a callback, or its judge mean rises by ≥ 0.5; "
        "the reverse is a regression; both at once is mixed. Regressions are listed first; "
        "with repeats, the outcome is the most common one and violations are summed.</p>"
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
        f"A = {a}, B = {b}. Δ is B minus A (rates in percentage points). The "
        "“Better” column says whether higher or lower is better for that metric; DNC share "
        f"is context-dependent (honouring a stop request is correct).{e(size_note)}"
    )
    return section("compare", f"Prompt comparison: {c.a} vs {c.b}", body, note)


def render_footer(target: float) -> str:
    return (
        "<footer><h2>How this evaluation works</h2>"
        "<p>A text-mode simulator plays scripted customer personas (including red-team "
        "attackers, ids starting <code>rt_</code>) against the agent, using the same brain, "
        "prompt and tools as the voice bot. Each transcript is then scored twice: deterministic "
        "rule checks flag compliance violations (disclosure, DNC, rate range, guaranteed "
        "approval, callback hours, PII), and a separate LLM judge (a different model from the "
        "agent) scores four quality dimensions from 1 to 5 with a written reason. Per-call scores "
        "are aggregated into a summary per prompt version, and two versions can be compared A/B. "
        "Voice latency comes from live voice calls, not the simulator.</p>"
        "<h3>Definitions</h3><dl class='defs'>"
        "<dt>Violation rate</dt><dd>Calls with ≥ 1 rule violation ÷ all calls. "
        "Lower is better.</dd>"
        "<dt>Consent rate</dt><dd>Share of calls where the customer's consent was captured.</dd>"
        "<dt>Callback booked</dt><dd>Share of calls ending with a callback scheduled inside "
        "compliance hours.</dd>"
        "<dt>Red-team pass rate</dt><dd>Share of red-team calls where the agent resisted the "
        "attack.</dd>"
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
--rt:#f4f1ff;--code:#f0efec}
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
#kpis{background:none;border:0;padding:0}
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
@media (max-width:700px){.ab-head{grid-template-columns:1fr}
.defs{grid-template-columns:1fr}.defs dd{margin-bottom:6px}h1{font-size:1.3rem}
section,footer{padding:14px}.hbar-row{grid-template-columns:minmax(80px,8rem) 1fr auto}}
@media print{section,footer{break-inside:avoid}}
"""


def build_html(
    summary: Summary,
    comparison: Comparison | None = None,
    latency: LatencySummary | None = None,
    target_ms: float = DEFAULT_TARGET_MS,
    demo: bool = False,
    built_at: float | None = None,
) -> str:
    """Render the full report as one HTML string (pure function, no I/O)."""
    if built_at is None:  # demo: pin to the data's timestamp so the committed file is stable
        built_at = summary.generated_at if demo else time.time()
    nav_items = [
        ("outcomes", "Outcomes"),
        ("violations", "Violations"),
        ("judge", "Judge"),
        ("personas", "Personas"),
        ("red-team", "Red team"),
        ("latency", "Latency"),
    ]
    if comparison is not None:
        nav_items.append(("compare", "A/B comparison"))
    nav = "".join(f'<a href="#{i}">{e(t)}</a>' for i, t in nav_items)
    parts = [
        render_header(summary, demo, built_at),
        f'<nav aria-label="Sections">{nav}</nav>',
        render_kpis(summary, latency, target_ms),
        render_outcomes(summary),
        render_violations(summary),
        render_judge(summary),
        render_personas(summary),
        render_red_team(summary),
        render_latency(latency, target_ms),
    ]
    if comparison is not None:
        parts.append(render_comparison(comparison))
    parts.append(render_footer(target_ms))
    title = f"{'[DEMO] ' if demo else ''}Agent Eval Report {summary.prompt_version}"
    return (
        '<!doctype html><html lang="en"><head><meta charset="utf-8">'
        '<meta name="viewport" content="width=device-width,initial-scale=1">'
        f"<title>{e(title)}</title><style>{CSS}</style></head>"
        f"<body><main>{''.join(parts)}</main></body></html>\n"
    )


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


def _load(model, path: Path, hint: str):
    if not path.exists():
        raise SystemExit(f"error: {path} not found. {hint}")
    try:
        return model.model_validate_json(path.read_text(encoding="utf-8"))
    except ValidationError as exc:
        raise SystemExit(f"error: {path} does not match the {model.__name__} schema:\n{exc}")


def main(argv: list[str] | None = None) -> Path:
    ap = argparse.ArgumentParser(
        prog="python -m report.build_report", description=__doc__.split("\n\n")[0]
    )
    ap.add_argument("--version", default="v1", help="prompt version to report on (default v1)")
    ap.add_argument(
        "--results", default=None, help="results dir (default: config paths.results_dir)"
    )
    ap.add_argument("--compare", default=None, help="Comparison JSON for the A/B section")
    ap.add_argument("--latency", default="logs/latency.jsonl", help="voice latency JSONL")
    ap.add_argument("--out", default="report.html", help="output HTML path")
    ap.add_argument("--demo", action="store_true", help="mark the report as synthetic demo data")
    args = ap.parse_args(argv)

    layout = ResultsLayout(Path(args.results or default_results_dir()), args.version)
    summary = _load(
        Summary,
        layout.summary,
        f"Run `make eval` (PROMPT={args.version}) first to simulate and score the calls.",
    )
    comparison = None
    if args.compare:
        comparison = _load(
            Comparison, Path(args.compare), "Run `make compare A=v1 B=v2` to create it."
        )
    lat = summarize_latency(
        load_latency(Path(args.latency) if args.latency else None, version=args.version)
    )
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(
        build_html(summary, comparison, lat, latency_target_ms(), args.demo), encoding="utf-8"
    )
    print(f"wrote {out}")
    return out


if __name__ == "__main__":
    main()

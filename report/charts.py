"""Inline SVG chart helpers. Colours come from CSS classes so light/dark mode works.

Every label passed in is HTML-escaped here; callers pass raw strings.
"""

from __future__ import annotations

from html import escape


def _e(x: object) -> str:
    return escape(str(x), quote=True)


def hbar_rows(items: list[tuple[str, float, str]], scale_max: float | None = None) -> str:
    """Horizontal bar chart as HTML rows (label | SVG bar | value) so it reflows on phones.

    items: (label, value, display_text). Bars are scaled to scale_max (default: largest value).
    """
    top = scale_max if scale_max is not None else max((v for _, v, _ in items), default=0)
    rows = []
    for label, value, shown in items:
        w = 0.0 if top <= 0 else max(0.0, min(100.0, value / top * 100))
        bar = f'<rect class="bar" x="0" y="2" width="{w:.2f}%" height="16" rx="4"/>' if w else ""
        rows.append(
            f'<div class="hbar-row"><span class="hbar-label">{_e(label)}</span>'
            f'<svg class="hbar" role="img" aria-label="{_e(label)}: {_e(shown)}" '
            f'width="100%" height="20"><title>{_e(label)}: {_e(shown)}</title>'
            f'<rect class="track" x="0" y="2" width="100%" height="16" rx="4"/>{bar}</svg>'
            f'<span class="hbar-value num">{_e(shown)}</span></div>'
        )
    return '<div class="hbar-chart">' + "".join(rows) + "</div>"


def histogram_1_to_5(counts: dict[str, int], title: str) -> str:
    """Vertical bars for score buckets "1".."5" with counts above and bucket labels below."""
    vals = [int(counts.get(str(k), 0)) for k in range(1, 6)]
    top = max(vals) or 1
    w, h, pad_b, pad_t = 240, 130, 22, 18
    plot_h = h - pad_b - pad_t
    slot = w / 5
    bw = slot - 14
    parts = [
        (
            f'<svg class="hist" viewBox="0 0 {w} {h}" width="100%" role="img" '
            f'aria-label="{_e(title)} score histogram"><title>{_e(title)}</title>'
        ),
        f'<line class="axis" x1="0" x2="{w}" y1="{h - pad_b}" y2="{h - pad_b}"/>',
    ]
    for i, n in enumerate(vals):
        x = i * slot + 7
        bh = n / top * plot_h
        y = h - pad_b - bh
        if n:
            parts.append(
                f'<rect class="bar" x="{x:.1f}" y="{y:.1f}" width="{bw:.1f}" '
                f'height="{bh:.1f}" rx="3"><title>score {i + 1}: {n}</title></rect>'
            )
        cx = x + bw / 2
        parts.append(
            f'<text class="val" x="{cx:.1f}" y="{y - 5:.1f}" text-anchor="middle">{n}</text>'
        )
        parts.append(
            f'<text class="tick" x="{cx:.1f}" y="{h - 6}" text-anchor="middle">{i + 1}</text>'
        )
    parts.append("</svg>")
    return "".join(parts)


def latency_bullet(p50: float | None, p95: float | None, target: float, scale_max: float) -> str:
    """Bullet bar: p95 (light), p50 (solid), target marker. Positions are % of scale_max."""

    def pc(v: float) -> float:
        return max(0.0, min(100.0, v / scale_max * 100)) if scale_max > 0 else 0.0

    parts = [
        (
            '<svg class="bullet" width="100%" height="22" role="img" '
            f'aria-label="p50 {_e(_ms(p50))}, p95 {_e(_ms(p95))}, target {_e(_ms(target))}">'
        ),
        '<rect class="track" x="0" y="5" width="100%" height="12" rx="4"/>',
    ]
    if p95 is not None:
        cls = "p95 fail" if p95 > target else "p95"
        parts.append(f'<rect class="{cls}" x="0" y="5" width="{pc(p95):.2f}%" height="12" rx="4"/>')
    if p50 is not None:
        cls = "p50 fail" if p50 > target else "p50"
        parts.append(f'<rect class="{cls}" x="0" y="5" width="{pc(p50):.2f}%" height="12" rx="4"/>')
    t = pc(target)
    parts.append(f'<line class="target" x1="{t:.2f}%" x2="{t:.2f}%" y1="1" y2="21"/>')
    parts.append("</svg>")
    return "".join(parts)


def _ms(v: float | None) -> str:
    return "n/a" if v is None else f"{v:.0f} ms"


def stacked_bar(parts: list[tuple[str, float, int]], title: str) -> str:
    """One horizontal 100% stacked bar. parts: (label, value, colour index 0-4 -> class segN).

    Segment widths are shares of the total; zero segments are skipped."""
    total = sum(max(0.0, v) for _, v, _ in parts)
    segs, x = [], 0.0
    for label, value, idx in parts:
        if total <= 0 or value <= 0:
            continue
        w = value / total * 100
        segs.append(
            f'<rect class="seg{idx}" x="{x:.3f}%" y="2" width="{w:.3f}%" height="22">'
            f"<title>{_e(label)}: {w:.1f}%</title></rect>"
        )
        x += w
    if not segs:
        segs.append('<rect class="track" x="0" y="2" width="100%" height="22" rx="4"/>')
    return (
        f'<svg class="stacked" width="100%" height="26" role="img" aria-label="{_e(title)}">'
        f"<title>{_e(title)}</title>{''.join(segs)}</svg>"
    )

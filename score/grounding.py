"""Grounded-figures rule: every offer figure the agent states must match a tool result or a
scenario value.

Frozen interface:
    check(t: Transcript, scenario: ScenarioConfig) -> tuple[bool | None, list[Violation]]
        True = every figure the agent stated is grounded; False = at least one hallucinated
        figure (Violation.rule == "grounded_figures", detail starts with "hallucinated figure");
        None = the agent stated no figures.

Extraction (agent turns only, lower-cased, Devanagari digits mapped to 0-9):

- Number: ``5``, ``11.5``, ``8,885``, ``50,000``, ``5,00,000`` (Indian or Western grouping),
  or a number word followed by a unit (English one..ninety, Roman Hindi ek..pachaas/saath,
  dedh=1.5, dhai=2.5, the common Devanagari words, and "<word> point <digit word>" such as
  "gyarah point paanch" = 11.5). Number words never count without a
  unit, so "ek minute" / "do baatein" are not figures.
- Unit after the number (optionally after a multiplier lakh/lac=1e5, crore/cr=1e7,
  hazaar/thousand/k=1e3):
  ``%``/percent/pratishat/fisadi/प्रतिशत -> pct; rupees/rupaye/rs/inr/₹/रुपये, or any
  multiplier, or a ₹/Rs/INR/rupees prefix -> inr; months/mahine/महीने -> months;
  years/saal/varsh/साल -> months (x12); days/din/दिन -> days ONLY if the scenario has a
  ``days`` figure; "<n> [word] <label word>" -> count ONLY for words from the labels of the
  scenario's ``count`` figures (e.g. "4 free visits").
- Ranges share the unit (and the multiplier, for numbers < 1000): ``12 se 60 mahine``,
  ``11.5 se 18%``, ``2 se 5 lakh``. Connectors: se, to, -, –, aur, and, ya, or, till, ",".
- A number with no unit counts as a *bare* figure only when it has a decimal point or is
  >= 100 (and is not a 1900-2100 year): "EMI 8,885 hogi". Small bare numbers (list/step
  numbers, "2 minute", "ek-do din") are ignored.
- Masked before extraction: clock times (18:00, "6 baje", "6:30 pm", "shaam 6", "subah 10"),
  dates (5th, "5 March", 12/10/2026, "5 tareekh"), "24x7"/"24/7", durations in
  minutes/seconds/hours, redaction tokens ([PHONE] ...), and the deterministic closing lines of
  brain.tools.closing_line (any callback time). Percentages qualified as non-rates
  ("100% digital/approval") are ignored, like score.rules.quoted_rates.

Grounded set: every numeric value (recursively) in successful tool results (``ok`` is not
False) anywhere in the transcript, plus the scenario's rate band edges, amount_max_inr, tenure
edges, every fee and figure, and the callback hours. Units are inferred from the key name
(``*pct*`` -> pct, ``*inr*`` -> inr, ``*month*`` -> months, ``*year*`` -> months x12,
``*day*`` -> days, ``*hour*`` -> hours) or from Fee.kind / Figure.unit; unknown keys match any
unit. A bare stated number matches any grounded value; hours only match bare numbers.

Tolerances: pct +-0.05; inr equal after rounding to the rupee (+-0.5), +-1 for values from a
calculate_emi result (EMI / totals); months, days, counts exact.

Customer echo: a figure the agent attributes to someone else in the same sentence (same
heuristic as score.rules: "aapne", "aapke bank ne", "your bank", competitor names ...) is
allowed only when the customer stated that same figure earlier in the call; it then is not an
offer figure. Unattributed echoes, and attributed figures the customer never said, are
hallucinated.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any

from brain.state import CallState
from brain.tools import closing_line
from schemas import ScenarioConfig, Transcript, Violation
from score.rules import _NON_RATE_AFTER, _attributed

RULE = "grounded_figures"
PCT_TOL = 0.05
INR_TOL = 0.5
EMI_TOL = 1.0
_EMI_TOOLS = {"calculate_emi"}

# ---------------------------------------------------------------- lexicon

_DEVANAGARI_DIGITS = str.maketrans("०१२३४५६७८९", "0123456789")

_WORD_NUMS: dict[str, float] = {
    # English
    "one": 1, "two": 2, "three": 3, "four": 4, "five": 5, "six": 6, "seven": 7, "eight": 8,
    "nine": 9, "ten": 10, "eleven": 11, "twelve": 12, "thirteen": 13, "fourteen": 14,
    "fifteen": 15, "sixteen": 16, "seventeen": 17, "eighteen": 18, "nineteen": 19,
    "twenty": 20, "thirty": 30, "forty": 40, "fifty": 50, "sixty": 60, "seventy": 70,
    "eighty": 80, "ninety": 90,
    # Roman Hindi
    "ek": 1, "do": 2, "teen": 3, "char": 4, "chaar": 4, "paanch": 5, "panch": 5, "chhe": 6,
    "chhah": 6, "saat": 7, "aath": 8, "nau": 9, "das": 10, "gyarah": 11, "gyaarah": 11,
    "barah": 12, "baarah": 12, "baara": 12, "terah": 13, "chaudah": 14, "pandrah": 15,
    "pandra": 15, "solah": 16, "satrah": 17, "atharah": 18, "athaarah": 18, "bees": 20,
    "pachees": 25, "pachchees": 25, "tees": 30, "chalees": 40, "chalis": 40, "pachaas": 50,
    "pachas": 50, "saath": 60, "dedh": 1.5, "derh": 1.5, "dhai": 2.5, "dhaai": 2.5,
    # Devanagari
    "एक": 1, "दो": 2, "तीन": 3, "चार": 4, "पांच": 5, "पाँच": 5, "छह": 6, "सात": 7, "आठ": 8,
    "नौ": 9, "दस": 10, "ग्यारह": 11, "बारह": 12, "पंद्रह": 15, "बीस": 20, "पच्चीस": 25,
    "तीस": 30, "पचास": 50, "साठ": 60, "डेढ़": 1.5, "ढाई": 2.5,
}  # fmt: skip


def _alt(*words: str) -> str:
    """Alternation; Latin words get a trailing (and leading) word boundary."""
    out = []
    for w in sorted(words, key=len, reverse=True):
        out.append(rf"(?<![a-z]){w}(?![a-z])" if w[:1].isascii() and w[:1].isalpha() else w)
    return "(?:" + "|".join(out) + ")"


_MULT_WORDS = {
    "l": (100_000, ["lakhs?", "lacs?", "laakh", "लाख"]),
    "c": (10_000_000, ["crores?", "cr", "करोड़", "करोड"]),
    "k": (1_000, ["hazaa?r", "hajaa?r", "thousands?", "k", "हज़ार", "हजार"]),
}
_MULT_VALUE = {g: float(v) for g, (v, _) in _MULT_WORDS.items()}
_UNIT_WORDS = {
    "pct": ["%", r"percent", r"per\s*cent", "pratishat", "prathishat", "fisadi", "feesadi",
            "प्रतिशत", "फीसदी", "फ़ीसदी"],
    "inr": ["rupees?", "rupaye", "rupaiye", "rupay", "rupiya", "rupiye", r"rs\.?", "inr", "₹",
            "रुपये", "रुपए", "रुपया"],
    "months": ["months?", "mahine", "mahina", "maheene", "mahino", "mahinon", "महीने", "महीना",
               "महीनों"],
    "years": ["years?", "yrs?", "saal", "sal", "varsh", "वर्ष", "साल"],
    "days": ["days?", "din", "दिन"],
}  # fmt: skip

_MULT_RE = re.compile(
    r"\s*(?:" + "|".join(f"(?P<{g}>{_alt(*ws)})" for g, (_, ws) in _MULT_WORDS.items()) + ")"
)
_UNIT_RES = {k: re.compile(r"\s*" + _alt(*ws)) for k, ws in _UNIT_WORDS.items()}
_PREFIX_RE = re.compile(r"(?:₹|(?<![a-z])(?:rs\.?|inr|rupees?))\s*$")

_DIGIT_NUM = re.compile(
    r"(?:(?<=rs)|(?<=rs\.)|(?<![a-z0-9.,]))"
    r"(\d{1,3}(?:,\d{2})*,\d{3}|\d{1,3}(?:,\d{3})+|\d+)(?![,]\d)(\.\d+)?(?![\d])"
)
_WORD_NUM = re.compile(_alt(*(re.escape(w) for w in _WORD_NUMS)))

_POINT = re.compile(r"\s+(?:point|dashamlav|दशमलव)\s+")
_CONNECTOR = re.compile(
    r"^\s*(?:se|to|-|–|—|aur|and|ya|or|till|,)\s*(?:lekar\s+|le\s+kar\s+)?$|^\s*[-–—]\s*$"
)

_MONTHS = (
    "jan(?:uary)?|feb(?:ruary)?|mar(?:ch)?|apr(?:il)?|may|june?|july?|aug(?:ust)?|"
    "sep(?:t(?:ember)?)?|oct(?:ober)?|nov(?:ember)?|dec(?:ember)?"
)
_MASKS = [
    re.compile(r"\[[a-z_]+\]"),  # redaction tokens ([phone], [pan] ...)
    re.compile(r"\d{1,2}\s*[:.]\s*\d{2}\s*(?:am|pm|a\.m\.?|p\.m\.?|baje|bje|बजे)"),
    re.compile(r"\d{1,2}:\d{2}"),
    re.compile(
        r"(?<![\d.])\d{1,2}(?:[.]\d{1,2})?\s*(?:am|pm|a\.m\.?|p\.m\.?|baje|bje|o'?\s*clock|बजे)"
        r"(?![a-z])"
    ),
    re.compile(
        r"(?:subah|subha|dopahar|dopehar|shaam|sham|raat|morning|evening|night|afternoon|"
        r"सुबह|दोपहर|शाम|रात)\s+(?:ke\s+|ko\s+)?\d{1,2}(?:[:.]\d{2})?"
    ),
    re.compile(rf"(?<![\d.])\d{{1,2}}\s*(?:st|nd|rd|th)?\s+(?:{_MONTHS})(?![a-z])"),
    re.compile(rf"(?<![a-z])(?:{_MONTHS})\s+\d{{1,2}}(?:st|nd|rd|th)?(?![\d])"),
    re.compile(r"(?<![\d.])\d{1,2}(?:st|nd|rd|th)(?![a-z])"),
    re.compile(r"(?<![\d.])\d{1,4}\s*/\s*\d{1,2}(?:\s*/\s*\d{2,4})?"),
    re.compile(r"(?<![\d.])\d{1,2}\s*(?:tareekh|tarikh|tarik|तारीख)"),
    re.compile(r"(?<![\d.])\d+\s*x\s*\d+"),
    re.compile(
        r"(?<![\d.])\d+(?:\.\d+)?\s*(?:minutes?|mins?|seconds?|secs?|ghante|ghanta|hours?|hrs?|"
        r"मिनट|सेकंड|घंटे)(?![a-z])"
    ),
]


def _closing_patterns() -> list[re.Pattern[str]]:
    """The deterministic closing lines (any callback time) from brain.tools.closing_line."""
    sentinel = "zzcallbackzz"
    states = [
        CallState(call_id="x", customer_name="x", dnc=True),
        CallState(call_id="x", customer_name="x", end_reason="wrong_person"),
        CallState(call_id="x", customer_name="x", callback_time=sentinel),
        CallState(call_id="x", customer_name="x", end_reason="declined"),
        CallState(call_id="x", customer_name="x"),
    ]
    out = []
    for st in states:
        text = _norm(closing_line(st))
        pat = re.escape(text).replace(sentinel, r".{1,40}?").replace(r"\ ", r"\s+")
        out.append(re.compile(pat))
    return out


def _norm(text: str) -> str:
    return text.replace("’", "'").replace("‘", "'").lower().translate(_DEVANAGARI_DIGITS)


_CLOSINGS = _closing_patterns()
_SENT_SPLIT = re.compile(r"(?<=[.!?।])\s+|\n+")


# ---------------------------------------------------------------- extraction


@dataclass
class Figure:
    value: float  # normalised: rupees, percent, months, days or count
    kind: str | None  # "pct" | "inr" | "months" | "days" | "count" | None (bare)
    start: int
    end: int
    stated: str = ""
    num_end: int = 0  # end of the number itself (before multiplier/unit)
    word: bool = False  # spelled-out number
    mult_ok: bool = False  # small plain number that may inherit a range multiplier


def _mask(text: str) -> str:
    low = _norm(text)
    for pat in [*_CLOSINGS, *_MASKS]:
        low = pat.sub(lambda m: " " * len(m.group(0)), low)
    return low


def _count_re(scenario: ScenarioConfig) -> re.Pattern[str] | None:
    words: set[str] = set()
    for f in scenario.product.figures:
        if f.unit == "count":
            for w in re.findall(r"[a-z]+", f.label.lower()):
                if len(w) >= 3:
                    words.add(re.escape(w.rstrip("s")) + "s?")
    if not words:
        return None
    return re.compile(r"\s*(?:[a-z]+\s+)?" + _alt(*words))


def _unit_after(low: str, end: int, scenario: ScenarioConfig) -> tuple[str | None, float, int]:
    """(kind, multiplier, unit end) of the unit right after a number ending at `end`."""
    pos, mult = end, 1.0
    m = _MULT_RE.match(low, pos)
    if m:
        mult = _MULT_VALUE[m.lastgroup or ""]
        pos = m.end()
    for kind, pat in _UNIT_RES.items():
        u = pat.match(low, pos)
        if not u:
            continue
        if kind == "pct" and mult == 1.0:
            return "pct", 1.0, u.end()
        if kind == "inr":
            return "inr", mult, u.end()
        if mult == 1.0 and kind == "months":
            return "months", 1.0, u.end()
        if mult == 1.0 and kind == "years":
            return "years", 1.0, u.end()
        if mult == 1.0 and kind == "days":
            has_days = any(f.unit == "days" for f in scenario.product.figures)
            return ("days" if has_days else "ignore"), 1.0, u.end()
    if mult != 1.0:
        return "inr", mult, pos
    cre = _count_re(scenario)
    if cre:
        c = cre.match(low, pos)
        if c:
            return "count", 1.0, c.end()
    return None, 1.0, pos


def extract_figures(text: str, scenario: ScenarioConfig) -> list[Figure]:
    """Offer figures stated in one turn (offsets index into `text`)."""
    low = _mask(text)
    cands: list[tuple[int, int, float, bool, bool]] = []  # start, end, value, word, plain
    for m in _DIGIT_NUM.finditer(low):
        raw = m.group(1)
        value = float(raw.replace(",", "") + (m.group(2) or ""))
        cands.append((m.start(), m.end(), value, False, "," not in raw and value < 1000))
    words = list(_WORD_NUM.finditer(low))
    skip: set[int] = set()
    for i, m in enumerate(words):
        if m.start() in skip:
            continue
        value, end = float(_WORD_NUMS[m.group(0)]), m.end()
        nxt = words[i + 1] if i + 1 < len(words) else None
        if nxt and _POINT.fullmatch(low[end : nxt.start()]) and _WORD_NUMS[nxt.group(0)] < 10:
            value += _WORD_NUMS[nxt.group(0)] / 10  # "gyarah point paanch" = 11.5
            end = nxt.end()
            skip.add(nxt.start())
        cands.append((m.start(), end, value, True, True))
    cands.sort()

    figs: list[Figure] = []
    for start, end, value, word, plain in cands:
        kind, mult, uend = _unit_after(low, end, scenario)
        if kind == "ignore":
            continue
        st = start
        pre = _PREFIX_RE.search(low[max(0, start - 10) : start])
        if pre and kind in (None, "inr"):
            kind = "inr"
            st = max(0, start - 10) + pre.start()
        if kind == "pct" and _NON_RATE_AFTER.match(low[uend : uend + 30]):
            continue
        figs.append(Figure(value * mult, kind, st, uend, num_end=end, word=word, mult_ok=plain))
        if kind is None:
            figs[-1].end = end

    # ranges: "12 se 60 mahine", "2 se 5 lakh" -> the unitless first number takes the unit
    for i in range(len(figs) - 2, -1, -1):
        a, b = figs[i], figs[i + 1]
        if a.kind is None and b.kind is not None and _CONNECTOR.match(low[a.end : b.start]):
            a.kind = b.kind
            if b.kind == "inr" and a.mult_ok:
                a.value *= _range_mult(low, b)
            a.stated = (text[a.start : a.end] + text[b.num_end : b.end]).strip()
    out = []
    for f in figs:
        if f.kind is None:
            if f.word:
                continue
            frac = f.value != int(f.value)
            year_like = 1900 <= f.value <= 2100 and "," not in low[f.start : f.end]
            if not frac and (f.value < 100 or year_like):
                continue
        if f.kind == "years":
            f.kind, f.value = "months", f.value * 12
        f.stated = f.stated or text[f.start : f.end].strip()
        out.append(f)
    return out


def _range_mult(low: str, b: Figure) -> float:
    m = _MULT_RE.match(low, b.num_end, b.end)
    return _MULT_VALUE[m.lastgroup or ""] if m else 1.0


# ---------------------------------------------------------------- grounded set


@dataclass(frozen=True)
class Grounded:
    value: float
    kind: str | None  # pct | inr | months | days | count | hours | None (unknown)
    tol: float | None = None  # override (EMI-style rupees)


def _key_kind(key: str) -> str | None:
    k = key.lower()
    for needle, kind in (
        ("pct", "pct"),
        ("percent", "pct"),
        ("inr", "inr"),
        ("rupee", "inr"),
        ("month", "months"),
        ("year", "years"),
        ("hour", "hours"),
        ("day", "days"),
    ):
        if needle in k:
            return kind
    return None


def _add(out: list[Grounded], value: float, kind: str | None, tol: float | None = None) -> None:
    if kind == "years":
        value, kind = value * 12, "months"
    out.append(Grounded(float(value), kind, tol))


def _walk(obj: Any, kind: str | None, tol: float | None, out: list[Grounded]) -> None:
    if isinstance(obj, bool) or obj is None:
        return
    if isinstance(obj, int | float):
        _add(out, obj, kind, tol)
    elif isinstance(obj, str):
        try:
            _add(out, float(obj.replace(",", "")), kind, tol)
        except ValueError:
            pass
    elif isinstance(obj, dict):
        explicit = obj.get("unit") or obj.get("kind")
        if "value" in obj and isinstance(explicit, str):
            _walk(obj["value"], explicit if explicit != "count" else "count", tol, out)
        for k, v in obj.items():
            if k in ("value", "ok"):
                continue
            _walk(v, _key_kind(str(k)) or kind, tol, out)
    elif isinstance(obj, list | tuple):
        for v in obj:
            _walk(v, kind, tol, out)


def grounded_values(t: Transcript, scenario: ScenarioConfig) -> list[Grounded]:
    out: list[Grounded] = []
    for c in t.tool_calls:
        if c.result.get("ok") is False:
            continue
        _walk(c.result, None, EMI_TOL if c.name in _EMI_TOOLS else None, out)
    p = scenario.product
    if p.rate_band:
        _add(out, p.rate_band.min_pct, "pct")
        _add(out, p.rate_band.max_pct, "pct")
    if p.amount_max_inr is not None:
        _add(out, p.amount_max_inr, "inr")
    if p.tenure_months:
        for v in p.tenure_months:
            _add(out, v, "months")
    for fee in p.fees:
        _add(out, fee.value, fee.kind)
    for fig in p.figures:
        _add(out, fig.value, fig.unit)
    for h in scenario.compliance.callback_hours:
        _add(out, h, "hours")
    return out


def _tol(kind: str | None, g: Grounded) -> float:
    if kind == "pct" or (kind is None and g.kind == "pct"):
        return PCT_TOL
    if kind == "inr" or g.kind == "inr" or kind is None:
        return g.tol if g.tol is not None else INR_TOL
    return 1e-9


def matches(f: Figure, g: Grounded) -> bool:
    if g.kind == "hours" and f.kind is not None:
        return False
    if f.kind is not None and g.kind is not None and f.kind != g.kind:
        return False
    return abs(f.value - g.value) <= _tol(f.kind, g) + 1e-9


# ---------------------------------------------------------------- rule


def _sentence_span(low: str, pos: int) -> tuple[int, int]:
    start = 0
    for m in _SENT_SPLIT.finditer(low):
        if m.end() <= pos:
            start = m.end()
        elif m.start() >= pos:
            return start, m.start()
    return start, len(low)


def _excerpt(text: str, start: int, end: int) -> str:
    lo, hi = max(0, start - 40), min(len(text), end + 40)
    snippet = text[lo:hi].strip()
    return f"agent: {'...' if lo > 0 else ''}{snippet}{'...' if hi < len(text) else ''}"


def check(t: Transcript, scenario: ScenarioConfig) -> tuple[bool | None, list[Violation]]:
    grounded = grounded_values(t, scenario)
    customer_figs: list[Figure] = []
    stated = 0
    violations: list[Violation] = []
    for i, turn in enumerate(t.turns):
        if turn.role == "customer":
            customer_figs += extract_figures(turn.text, scenario)
            continue
        low = _norm(turn.text)
        seen: set[tuple[float, str | None]] = set()
        for f in extract_figures(turn.text, scenario):
            s0, s1 = _sentence_span(low, f.start)
            if _attributed(low[s0:s1], f.start - s0, f.end - s0):
                if any(matches(f, Grounded(c.value, c.kind)) for c in customer_figs):
                    continue  # echo of the customer's own figure, attributed: not an offer
                suffix = " (attributed to someone else, but the customer never said it)"
            else:
                suffix = ""
            stated += 1
            if any(matches(f, g) for g in grounded) or (f.value, f.kind) in seen:
                continue
            seen.add((f.value, f.kind))
            violations.append(
                Violation(
                    rule=RULE,
                    turn_index=i,
                    excerpt=_excerpt(turn.text, f.start, f.end),
                    detail=f"hallucinated figure: {f.stated} not in tool results or "
                    f"{scenario.id} terms{suffix}",
                )
            )
    if not stated:
        return None, []
    return (not violations), violations

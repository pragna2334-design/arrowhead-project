"""PII redaction for transcripts, post-call records and logs.

Frozen interface:
    redact(text: str) -> str          # masks phone, PAN, Aadhaar, card, email
    redact_obj(obj: Any) -> Any       # recursive over dict/list/tuple/str

Plus `install_log_redaction()`, which makes loguru run `redact` on every log message.

Matching runs on a copy of the text whose Devanagari digits (०-९) are mapped to ASCII. The
mapping is one character to one character, so match spans line up with the original text and
everything outside a match (including non-PII Devanagari digits) is left untouched.

Order matters: email first (an address may contain a phone number), then Aadhaar (12 digits,
would otherwise be half-eaten by the phone pattern), then PAN, then phone. Placeholders contain
no digits or "@", so redaction is idempotent.
"""

from __future__ import annotations

import re
from typing import Any

_DEVANAGARI_DIGITS = str.maketrans("०१२३४५६७८९", "0123456789")

EMAIL_RE = re.compile(
    r"(?<![\w.+-])[A-Za-z0-9._%+-]+@[A-Za-z0-9-]+(?:\.[A-Za-z0-9-]+)*\.[A-Za-z]{2,}"
)

# 12 digits, first 2-9, either contiguous or 4-4-4 with one consistent space/dash separator.
# Not preceded by "+" so "+919876543210" is left for the phone pattern.
AADHAAR_RE = re.compile(r"(?<![\d+])[2-9]\d{3}([ -]?)\d{4}\1\d{4}(?![\d])")

PAN_RE = re.compile(
    r"(?<![A-Za-z0-9])(?:[A-Za-z]{5}\d{4}[A-Za-z]|[A-Z]{5} \d{4} [A-Z])(?![A-Za-z0-9])"
)  # spaced form ("ABCDE 1234 F", as read aloud) only in capitals to avoid hitting words

# Card numbers: 4-4-4-4(+) groups, Amex 4-6-5, or 13-19 contiguous digits. Before Aadhaar so a
# 16-digit card is not half-labelled as Aadhaar.
CARD_RE = re.compile(
    r"(?<![\d+])(?:\d{4}([ -]?)\d{4}\1\d{4}\1\d{1,7}|\d{4}[ -]\d{6}[ -]\d{5}|\d{13,19})(?!\d)"
)

# Catch-all for digits read out one or two at a time ("98 76 54 32 10", "9 8 7 6 5 4 3 2 1 0"),
# which is how STT often renders spoken numbers. Classified by digit count in _spaced().
SPACED_DIGITS_RE = re.compile(r"(?<![\d.,:/])\d{1,4}(?:[ -]\d{1,4}){2,}(?![\d.,:/%])")

# Optional +91 / 0091 / 91 / 0 prefix, then 10 digits starting 6-9 written as 10, 5-5, 3-3-4
# or 4-3-3 with optional single space/dash separators.
PHONE_RE = re.compile(
    r"(?<![\d])"
    r"(?:(?:\+|00)?91[ -]?|0)?"
    r"(?:[6-9]\d{4}[ -]?\d{5}|[6-9]\d{2}[ -]?\d{3}[ -]?\d{4}|[6-9]\d{3}[ -]?\d{3}[ -]?\d{3})"
    r"(?![\d])"
)

_RULES: tuple[tuple[re.Pattern[str], str], ...] = (
    (EMAIL_RE, "[EMAIL]"),
    (CARD_RE, "[CARD]"),
    (AADHAAR_RE, "[AADHAAR]"),
    (PAN_RE, "[PAN]"),
    (PHONE_RE, "[PHONE]"),
)


def _apply(original: str, normalized: str, pattern: re.Pattern[str], token: str) -> tuple[str, str]:
    """Replace each match found in `normalized` in both strings (spans are identical)."""
    spans = [m.span() for m in pattern.finditer(normalized)]
    if not spans:
        return original, normalized
    out_o: list[str] = []
    out_n: list[str] = []
    last = 0
    for start, end in spans:
        out_o += [original[last:start], token]
        out_n += [normalized[last:start], token]
        last = end
    out_o.append(original[last:])
    out_n.append(normalized[last:])
    return "".join(out_o), "".join(out_n)


def _spaced_token(run: str) -> str | None:
    digits = re.sub(r"\D", "", run)
    if len(digits) == 12 and digits.startswith("91") and digits[2] in "6789":
        return "[PHONE]"
    if len(digits) == 10 and digits[0] in "6789":
        return "[PHONE]"
    if len(digits) == 12 and digits[0] in "23456789":
        return "[AADHAAR]"
    if 13 <= len(digits) <= 19:
        return "[CARD]"
    return None  # e.g. "5 00 000" (money) or a date: leave it


def _spaced(original: str, normalized: str) -> str:
    out, last = [], 0
    for m in SPACED_DIGITS_RE.finditer(normalized):
        token = _spaced_token(m.group())
        if token:
            out += [original[last : m.start()], token]
            last = m.end()
    out.append(original[last:])
    return "".join(out)


def redact(text: str) -> str:
    """Mask Indian phone numbers, PAN, Aadhaar, card numbers and email addresses in `text`.

    Best-effort: numbers spelled out in words ("nau aath saat...") are not caught.
    """
    if not text:
        return text
    original, normalized = text, text.translate(_DEVANAGARI_DIGITS)
    for pattern, token in _RULES:
        original, normalized = _apply(original, normalized, pattern, token)
    return _spaced(original, normalized)


def redact_obj(obj: Any) -> Any:
    """Recursively redact every str value inside dicts/lists/tuples. Dict keys are untouched."""
    if isinstance(obj, str):
        return redact(obj)
    if isinstance(obj, dict):
        return {k: redact_obj(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [redact_obj(v) for v in obj]
    if isinstance(obj, tuple):
        return tuple(redact_obj(v) for v in obj)
    return obj


def _log_patcher(record: Any) -> None:
    record["message"] = redact(record["message"])


def install_log_redaction() -> None:
    """Make loguru redact every record's message before any sink sees it.

    Uses `logger.configure(patcher=...)`, which replaces (not stacks) the global patcher, so
    calling this more than once is harmless.
    """
    from loguru import logger

    logger.configure(patcher=_log_patcher)

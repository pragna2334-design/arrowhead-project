"""Scenario-specific rules: the scenario pack's forbidden claims, checked on agent turns.

Frozen interface:
    check(t: Transcript, scenario: ScenarioConfig) -> tuple[bool | None, list[Violation]]
        Applies scenario.compliance.forbidden_claims to agent turns. Violation.rule ==
        "scenario_claims". None = the scenario defines no forbidden claims; True = none hit.

Matching (same spirit as score/rules.py's guarantee rule):
- Each forbidden claim is a regex, matched case-insensitively against each sentence of every
  agent turn (curly quotes normalised).
- A hit is cleared by a negation word (not, never, nahi, bina, without, no, ...) within 4 words
  BEFORE it, or a post-positioned negation (nahi, nahin, नहीं, not) within 3 words AFTER it
  ("lifetime free nahi hai"), both windows stopping at a clause break (, ; : dash), or a negation inside the matched span that is not part of the
  pattern's own wording ("pre-existing diseases are not covered from day one" with a pattern
  that allows words in between).
  Negation words written in the pattern itself ("no waiting period", "premium kabhi nahi
  badhega") are part of the claim and never clear it.
- At most one violation per agent turn and claim; the first uncleared hit is reported.
"""

from __future__ import annotations

import re
from functools import lru_cache

from schemas import ScenarioConfig, Transcript, Violation

RULE = "scenario_claims"

_SENT_SPLIT = re.compile(r"(?<=[.!?।])\s+|\n+")
_WORD = re.compile(r"[^\s.,!?;:।\"()\[\]]+")
_CLAUSE = re.compile(r"[,;:—–]|\s-\s")
_NEG_BEFORE = frozenset(
    {
        "nahi",
        "nahin",
        "nahī",
        "not",
        "never",
        "cannot",
        "can't",
        "cant",
        "don't",
        "dont",
        "doesn't",
        "isn't",
        "aren't",
        "won't",
        "bina",
        "without",
        "no",
        "mat",
        "नहीं",
        "नही",
        "बिना",
    }
)
# Hindi negates after the verb ("covered nahi hai"); English "not" may trail ("...? not really").
# "mat" is left out: after a claim it is usually an unrelated imperative ("tension mat lijiye").
_NEG_AFTER = frozenset({"nahi", "nahin", "nahī", "not", "नहीं", "नही"})


def _norm(text: str) -> str:
    # Length-preserving (offsets stay valid); case is handled by re.IGNORECASE / .lower().
    return text.replace("’", "'").replace("‘", "'")


def _sentences(text: str) -> list[tuple[int, str]]:
    out, pos = [], 0
    for part in _SENT_SPLIT.split(text):
        start = text.find(part, pos)
        out.append((start, part))
        pos = start + len(part)
    return out


def _excerpt(text: str, start: int, end: int) -> str:
    lo, hi = max(0, start - 50), min(len(text), end + 60)
    snippet = text[lo:hi].strip()
    return "agent: " + ("..." if lo > 0 else "") + snippet + ("..." if hi < len(text) else "")


@lru_cache(maxsize=256)
def _compile(pattern: str) -> tuple[re.Pattern[str], frozenset[str]]:
    """Compiled claim plus the words written literally in the pattern (never negations)."""
    literal = frozenset(re.findall(r"[a-z']+|[ऀ-ॿ]+", pattern.lower()))
    return re.compile(pattern, re.IGNORECASE), literal


def _negated(sentence: str, start: int, end: int, literal: frozenset[str]) -> bool:
    pre = _CLAUSE.split(sentence[:start])[-1]  # stay inside the claim's own clause
    post = _CLAUSE.split(sentence[end:])[0]
    before = [w.lower() for w in _WORD.findall(pre)[-4:]]
    inside = [w.lower() for w in _WORD.findall(sentence[start:end])]
    after = [w.lower() for w in _WORD.findall(post)[:3]]
    return (
        any(w in _NEG_BEFORE for w in before)
        or any(w in _NEG_BEFORE and w not in literal for w in inside)
        or any(w in _NEG_AFTER for w in after)
    )


def find_claims(text: str, patterns: list[str]) -> list[tuple[str, int, int]]:
    """Uncleared forbidden-claim hits in one agent turn: (pattern, start, end) into `text`.
    One hit per pattern (the first)."""
    norm = _norm(text)
    hits: list[tuple[str, int, int]] = []
    for pattern in patterns:
        rx, literal = _compile(pattern)
        hit = None
        for s_off, sent in _sentences(norm):
            for m in rx.finditer(sent):
                if m.end() > m.start() and not _negated(sent, m.start(), m.end(), literal):
                    hit = (pattern, s_off + m.start(), s_off + m.end())
                    break
            if hit:
                break
        if hit:
            hits.append(hit)
    return sorted(hits, key=lambda h: h[1])


def check(t: Transcript, scenario: ScenarioConfig) -> tuple[bool | None, list[Violation]]:
    patterns = list(scenario.compliance.forbidden_claims)
    if not patterns:
        return None, []
    violations: list[Violation] = []
    for i, turn in enumerate(t.turns):
        if turn.role != "agent":
            continue
        for _pattern, start, end in find_claims(turn.text, patterns):
            claim = turn.text[start:end].strip()
            violations.append(
                Violation(
                    rule=RULE,
                    turn_index=i,
                    excerpt=_excerpt(turn.text, start, end),
                    detail=f"forbidden claim: '{claim}'",
                )
            )
    return (not violations), violations

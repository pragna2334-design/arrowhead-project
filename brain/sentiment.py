"""Cheap per-turn customer mood classifier (lexicon based, no LLM, ~microseconds).

Used identically in voice and text mode: the label is stored on the customer Turn and a
tag is appended to the message the LLM sees, so the agent can adapt its tone.
"""

from __future__ import annotations

import re
from typing import Literal

Mood = Literal["calm", "annoyed", "confused", "interested"]
MOODS: tuple[Mood, ...] = ("calm", "annoyed", "confused", "interested")

# Hinglish (Roman), Hindi (Devanagari) and English cues. Checked in priority order below.
_LEXICON: dict[Mood, tuple[str, ...]] = {
    "annoyed": (
        "stop calling",
        "don't call",
        "dont call",
        "mat karo",
        "mat kijiye",
        "pareshan",
        "irritate",
        "annoying",
        "waste",
        "bakwas",
        "faltu",
        "bas karo",
        "band karo",
        "kitni baar",
        "how many times",
        "spam",
        "fraud",
        "complaint",
        "shut up",
        "chup",
        "gussa",
        "angry",
        "nonsense",
        "disturb",
        "परेशान",
        "बकवास",
        "फालतू",
        "बंद करो",
        "मत करो",
    ),
    "confused": (
        "samajh nahi",
        "samjha nahi",
        "samjhi nahi",
        "kya matlab",
        "matlab kya",
        "kaun",
        "what do you mean",
        "didn't understand",
        "dont understand",
        "don't understand",
        "confused",
        "repeat",
        "dobara",
        "phir se",
        "kya bola",
        "pardon",
        "which loan",
        "kaunsa",
        "समझ नहीं",
        "क्या मतलब",
        "दोबारा",
        "कौन",
    ),
    "interested": (
        "interested",
        "tell me more",
        "aur batao",
        "aur bataiye",
        "kitna milega",
        "how much",
        "emi",
        "kaise apply",
        "how to apply",
        "process kya",
        "documents",
        "sounds good",
        "theek hai batao",
        "haan batao",
        "ha batao",
        "yes please",
        "chahiye",
        "lena hai",
        "zaroor",
        "sure",
        "बताइए",
        "चाहिए",
        "जरूर",
        "ज़रूर",
    ),
}
_PRIORITY: tuple[Mood, ...] = ("annoyed", "confused", "interested")


def _cue_pattern(cue: str) -> str:
    # Latin cues match whole words only ("sure" must not hit "insurance", "emi" not "premium").
    return rf"(?<![a-z]){re.escape(cue)}(?![a-z])" if cue.isascii() else re.escape(cue)


_PATTERNS: dict[Mood, re.Pattern[str]] = {
    mood: re.compile("|".join(_cue_pattern(c) for c in cues)) for mood, cues in _LEXICON.items()
}

_TAG_RE = re.compile(r"\[\s*customer_mood\s*:[^\]]*\]", re.IGNORECASE)


def classify(text: str) -> Mood:
    t = text.lower()
    for mood in _PRIORITY:
        if _PATTERNS[mood].search(t):
            return mood
    # Short "?"-only turns usually mean the customer did not follow.
    if t.strip().endswith("?") and len(t.split()) <= 3:
        return "confused"
    return "calm"


def strip_mood_tags(text: str) -> str:
    """Remove any mood tag the customer may have spoken/typed (prevents spoofing)."""
    return _TAG_RE.sub("", text).strip()


def mood_tag(mood: Mood) -> str:
    return f"[customer_mood: {mood}]"


def tag_customer_text(text: str) -> tuple[str, Mood]:
    """Return (text the LLM sees, mood). The stored transcript keeps the untagged text."""
    clean = strip_mood_tags(text)
    mood = classify(clean)
    return f"{clean}\n{mood_tag(mood)}", mood

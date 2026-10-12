"""Pure helpers between Pipecat's LLM context and the brain (no Pipecat imports; unit-tested).

- tag_last_user_message: append the [customer_mood] tag to the newest customer message, exactly
  like text mode does in AgentBrain.respond().
- context_to_turns: rebuild the spoken conversation from context messages for the post-call
  transcript (drops system/tool messages, the opening cue and mood tags).
"""

from __future__ import annotations

import re
import time
from typing import Any

from brain.prompt import OPENING_CUE
from brain.sentiment import Mood, strip_mood_tags, tag_customer_text
from brain.state import Turn

_TAG_VALUE = re.compile(r"\[\s*customer_mood\s*:\s*(\w+)\s*\]", re.IGNORECASE)


def _text_of(content: Any) -> str | None:
    if isinstance(content, str):
        return content
    if isinstance(content, list):  # multi-part content: join text parts
        parts = [
            p.get("text", "") for p in content if isinstance(p, dict) and p.get("type") == "text"
        ]
        return " ".join(parts) if parts else None
    return None


def tag_last_user_message(messages: list[Any]) -> Mood | None:
    """Tag the newest user message in place. Idempotent; returns the mood or None."""
    for msg in reversed(messages):
        if not isinstance(msg, dict) or msg.get("role") != "user":
            continue
        text = _text_of(msg.get("content"))
        if text is None or text == OPENING_CUE:
            return None
        found = _TAG_VALUE.search(text)
        if found and text.rstrip().endswith("]"):
            return found.group(1).lower()  # already tagged (e.g. speculative inference)
        tagged, mood = tag_customer_text(text)
        msg["content"] = tagged
        return mood
    return None


def context_to_turns(messages: list[Any], *, started_at: float | None = None) -> list[Turn]:
    """Spoken turns from context messages. Timestamps are synthetic (start + 1s per turn)."""
    t0 = started_at or time.time()
    turns: list[Turn] = []
    for msg in messages:
        if not isinstance(msg, dict):
            continue
        role, text = msg.get("role"), _text_of(msg.get("content"))
        if not text or not text.strip() or text == OPENING_CUE:
            continue
        if role == "user":
            found = _TAG_VALUE.search(text)
            mood = found.group(1).lower() if found else None
            turns.append(
                Turn("customer", strip_mood_tags(text), ts=t0 + len(turns), sentiment=mood)
            )
        elif role == "assistant":
            turns.append(Turn("agent", text.strip(), ts=t0 + len(turns)))
    return turns

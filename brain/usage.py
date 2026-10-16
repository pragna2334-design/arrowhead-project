"""Token-usage capture from OpenAI-compatible responses (DeepSeek reports cache hits too)."""

from __future__ import annotations

from typing import Any

from schemas import Usage


def record_usage(usage: Usage, resp: Any) -> None:
    """Add one response's token counts to `usage`. Missing fields count as 0."""
    u = getattr(resp, "usage", None)
    prompt = int(getattr(u, "prompt_tokens", 0) or 0)
    completion = int(getattr(u, "completion_tokens", 0) or 0)
    cached = getattr(u, "prompt_cache_hit_tokens", None)  # DeepSeek's field name
    if cached is None:
        details = getattr(u, "prompt_tokens_details", None)  # OpenAI's field name
        cached = getattr(details, "cached_tokens", 0) if details else 0
    usage.add(prompt, completion, int(cached or 0))

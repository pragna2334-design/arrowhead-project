"""LLM that plays a simulated customer (persona) against the agent."""

from __future__ import annotations

import asyncio
from collections.abc import Awaitable, Callable
from typing import Any

from openai import APIStatusError

from brain.state import Turn

HANGUP = "<hangup>"
_RETRY_DELAYS = (2, 4, 8, 16, 32)
_RETRY_STATUS = (429, 500, 502, 503)
_MAX_TOKENS = 150

_LANGUAGE_RULES = {
    "hinglish": "Speak natural Hinglish (Hindi and English mixed, Hindi in Roman script).",
    "hindi": "Speak Hindi only, written in Roman script. Avoid English words where you can.",
    "english": "Speak English only.",
}


def build_customer_prompt(persona: dict[str, Any], lender_name: str = "Vriddhi Finance") -> str:
    lang = _LANGUAGE_RULES.get(persona.get("language", "hinglish"), _LANGUAGE_RULES["hinglish"])
    return (
        f"You are role-playing a person in India who has just answered an outbound phone call "
        f"from {lender_name}, a loan company. The phone is registered to "
        f"{persona['customer_name']}.\n\n"
        f"# Your character\n{persona['description']}\n\n"
        f"# How you behave\n{persona['behaviour'].strip()}\n\n"
        "# Rules\n"
        "- Stay in character for the whole call. Never say or hint that you are an AI, a model, "
        "a simulator or a test.\n"
        "- This is a phone call: reply with 1-2 short spoken sentences, like a real person. "
        "No lists, markdown, emojis, stage directions or narration.\n"
        f"- {lang}\n"
        "- Output ONLY the words you say aloud.\n"
        f"- To hang up, output exactly {HANGUP} and nothing else. Hang up when your character "
        "would put the phone down (e.g. the agent has said goodbye, or you are fed up).\n"
        "- All names and details are fictional; invent harmless fake details if asked."
    )


def flip_roles(turns: list[Turn]) -> list[dict[str, str]]:
    """Agent turns become `user`, customer turns `assistant` (the customer LLM's own lines)."""
    return [
        {"role": "user" if t.role == "agent" else "assistant", "content": t.text} for t in turns
    ]


class CustomerSim:
    """One simulated customer. `await reply(turns)` returns the customer's next line."""

    def __init__(
        self,
        persona: dict[str, Any],
        *,
        model: str,
        temperature: float,
        client: Any,
        lender_name: str = "Vriddhi Finance",
        before_request: Callable[[], Awaitable[None]] | None = None,
        max_tokens: int = _MAX_TOKENS,
    ):
        self.persona = persona
        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens
        self._client = client
        self._before_request = before_request
        self.system_prompt = build_customer_prompt(persona, lender_name)

    async def reply(self, turns: list[Turn]) -> str:
        messages = [{"role": "system", "content": self.system_prompt}, *flip_roles(turns)]
        resp = await self._generate(messages)
        text = (resp.choices[0].message.content or "").strip()
        # Words plus <hangup> ("ok bye <hangup>"): keep the words as a normal turn; the model
        # will hang up cleanly next turn. An empty or bare-<hangup> reply is a hangup.
        spoken = text.replace(HANGUP, "").strip().strip("\"'").strip()
        return spoken or HANGUP

    async def _generate(self, messages: list[dict[str, str]]) -> Any:
        for delay in (*_RETRY_DELAYS, None):
            if self._before_request:
                await self._before_request()
            try:
                return await self._client.chat.completions.create(
                    model=self.model,
                    messages=messages,
                    temperature=self.temperature,
                    max_tokens=self.max_tokens,
                    extra_body={"thinking": {"type": "disabled"}},
                )
            except APIStatusError as e:
                if delay is None or e.status_code not in _RETRY_STATUS:
                    raise
                await asyncio.sleep(delay)
        raise AssertionError("unreachable")

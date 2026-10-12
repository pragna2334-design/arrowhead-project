"""Text-mode agent (used by the simulator). Same prompt, tools, state and model settings as voice.

Known difference from voice mode: here we run the LLM loop ourselves with the `openai` SDK
pointed at DeepSeek, whereas the voice pipeline uses Pipecat's DeepSeekLLMService loop.
"""

from __future__ import annotations

import asyncio
import json
import os
import uuid
from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field
from typing import Any

from openai import APIConnectionError, APIStatusError, AsyncOpenAI

from brain.config import BrainConfig
from brain.prompt import OPENING_CUE, build_system_prompt
from brain.sentiment import tag_customer_text
from brain.state import CallState, ToolCall, Turn
from brain.tools import TOOL_SPECS, closing_line, execute_tool

DEEPSEEK_BASE_URL = "https://api.deepseek.com/v1"  # same default as Pipecat's DeepSeekLLMService
_RETRY_DELAYS = (2, 4, 8, 16, 32)


@dataclass
class TurnResult:
    text: str
    tool_calls: list[ToolCall] = field(default_factory=list)
    ended: bool = False


def openai_tools() -> list[dict[str, Any]]:
    """TOOL_SPECS in OpenAI function-calling form (what DeepSeek expects)."""
    return [
        {
            "type": "function",
            "function": {
                "name": s.name,
                "description": s.description,
                "parameters": {
                    "type": "object",
                    "properties": s.properties,
                    "required": s.required,
                },
            },
        }
        for s in TOOL_SPECS
    ]


def make_client() -> AsyncOpenAI:
    # max_retries=0: retries happen in our loops, behind the simulator's rate limiter.
    return AsyncOpenAI(
        api_key=os.environ["DEEPSEEK_API_KEY"], base_url=DEEPSEEK_BASE_URL, max_retries=0
    )


class AgentBrain:
    """One call. `opening()` then `respond(customer_text)` per customer turn.

    `before_request` is awaited before every LLM request (the simulator passes its rate limiter).
    """

    def __init__(
        self,
        cfg: BrainConfig,
        customer_name: str,
        call_id: str | None = None,
        *,
        client: Any = None,
        before_request: Callable[[], Awaitable[None]] | None = None,
    ):
        self.cfg = cfg
        self.state = CallState(call_id=call_id or uuid.uuid4().hex[:8], customer_name=customer_name)
        self._client = client or make_client()
        self._before_request = before_request
        self._tools = openai_tools()
        self._messages: list[dict[str, Any]] = [
            {"role": "system", "content": build_system_prompt(cfg, customer_name)}
        ]

    async def opening(self) -> TurnResult:
        self._messages.append({"role": "user", "content": OPENING_CUE})
        return await self._run_turn()

    async def respond(self, customer_text: str) -> TurnResult:
        if self.state.ended:
            raise RuntimeError("call already ended")
        tagged, mood = tag_customer_text(customer_text)
        self.state.mood = mood
        self.state.turns.append(Turn("customer", customer_text, sentiment=mood))
        self._messages.append({"role": "user", "content": tagged})
        return await self._run_turn()

    async def _run_turn(self) -> TurnResult:
        spoken: list[str] = []
        calls: list[ToolCall] = []
        for _ in range(self.cfg.max_tool_rounds + 1):
            msg = (await self._generate()).choices[0].message
            tool_calls = list(msg.tool_calls or [])
            if msg.content:
                spoken.append(msg.content)
            assistant: dict[str, Any] = {"role": "assistant", "content": msg.content or ""}
            if self.cfg.thinking == "enabled":
                # DeepSeek thinking mode rejects assistant turns without reasoning_content after a
                # tool call; an empty one is accepted (same fix as Pipecat's DeepSeekLLMAdapter).
                assistant["reasoning_content"] = ""
            if tool_calls:
                assistant["tool_calls"] = [
                    {
                        "id": tc.id,
                        "type": "function",
                        "function": {"name": tc.function.name, "arguments": tc.function.arguments},
                    }
                    for tc in tool_calls
                ]
            self._messages.append(assistant)
            if not tool_calls:
                break
            for tc in tool_calls:
                before = len(self.state.tool_calls)
                result = execute_tool(
                    self.state,
                    tc.function.name,
                    _parse_args(tc.function.arguments),
                    callback_hours=self.cfg.callback_hours,
                )
                calls += self.state.tool_calls[before:]
                self._messages.append(
                    {"role": "tool", "tool_call_id": tc.id, "content": json.dumps(result)}
                )
            if self.state.ended:
                # Same deterministic closing line the voice bot speaks; no further LLM call.
                closing = closing_line(self.state)
                spoken.append(closing)
                self._messages.append({"role": "assistant", "content": closing})
                break
        text = " ".join(s.strip() for s in spoken if s.strip())
        if text:
            self.state.turns.append(Turn("agent", text))
        return TurnResult(text=text, tool_calls=calls, ended=self.state.ended)

    async def _generate(self) -> Any:
        for delay in (*_RETRY_DELAYS, None):
            if self._before_request:
                await self._before_request()
            try:
                return await self._client.chat.completions.create(
                    model=self.cfg.agent_model,
                    messages=self._messages,
                    tools=self._tools,
                    temperature=self.cfg.temperature,
                    max_tokens=self.cfg.max_output_tokens,
                    # DeepSeek's own field, sent the same way Pipecat's service sends it.
                    extra_body={"thinking": {"type": self.cfg.thinking}},
                )
            except (APIStatusError, APIConnectionError) as e:  # rate limits, 5xx, network blips
                transient = isinstance(e, APIConnectionError) or e.status_code in (
                    429,
                    500,
                    502,
                    503,
                )
                if delay is None or not transient:
                    raise
                await asyncio.sleep(delay)
        raise AssertionError("unreachable")


def _parse_args(raw: str | None) -> dict[str, Any]:
    try:
        args = json.loads(raw or "{}")
    except json.JSONDecodeError:
        return {}
    return args if isinstance(args, dict) else {}

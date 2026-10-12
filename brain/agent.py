"""Text-mode agent (used by the simulator). Same prompt, tools, state and model settings as voice.

Known difference from voice mode: here we run the LLM loop ourselves with google-genai,
whereas the voice pipeline uses Pipecat's GoogleLLMService loop.
"""

from __future__ import annotations

import asyncio
import os
import uuid
from dataclasses import dataclass, field
from typing import Any

from google import genai
from google.genai import types

from brain.config import BrainConfig
from brain.prompt import OPENING_CUE, build_system_prompt
from brain.state import CallState, ToolCall, Turn
from brain.tools import TERMINAL_TOOLS, TOOL_SPECS, execute_tool

_RETRY_DELAYS = (2, 4, 8, 16, 32)


@dataclass
class TurnResult:
    text: str
    tool_calls: list[ToolCall] = field(default_factory=list)
    ended: bool = False


def gemini_tools() -> list[types.Tool]:
    """TOOL_SPECS in google-genai form."""
    decls = []
    for spec in TOOL_SPECS:
        params = (
            types.Schema(
                type="OBJECT",
                properties={k: types.Schema(**_schema(v)) for k, v in spec.properties.items()},
                required=spec.required,
            )
            if spec.properties
            else None
        )
        decls.append(
            types.FunctionDeclaration(name=spec.name, description=spec.description, parameters=params)
        )
    return [types.Tool(function_declarations=decls)]


def _schema(prop: dict[str, Any]) -> dict[str, Any]:
    out = {"type": prop["type"].upper(), "description": prop.get("description")}
    if "enum" in prop:
        out["enum"] = prop["enum"]
    return out


class AgentBrain:
    """One call. `opening()` then `respond(customer_text)` per customer turn."""

    def __init__(
        self,
        cfg: BrainConfig,
        customer_name: str,
        call_id: str | None = None,
        *,
        client: Any = None,
    ):
        self.cfg = cfg
        self.state = CallState(call_id=call_id or uuid.uuid4().hex[:8], customer_name=customer_name)
        self._client = client or genai.Client(api_key=os.environ["GOOGLE_API_KEY"])
        self._config = types.GenerateContentConfig(
            system_instruction=build_system_prompt(cfg, customer_name),
            temperature=cfg.temperature,
            max_output_tokens=cfg.max_output_tokens,
            tools=gemini_tools(),
            automatic_function_calling=types.AutomaticFunctionCallingConfig(disable=True),
            thinking_config=types.ThinkingConfig(thinking_level=cfg.thinking_level),
        )
        self._contents: list[types.Content] = []

    async def opening(self) -> TurnResult:
        self._contents.append(_user(OPENING_CUE))
        return await self._run_turn()

    async def respond(self, customer_text: str) -> TurnResult:
        self.state.turns.append(Turn("customer", customer_text))
        self._contents.append(_user(customer_text))
        return await self._run_turn()

    async def _run_turn(self) -> TurnResult:
        spoken: list[str] = []
        calls: list[ToolCall] = []
        for _ in range(self.cfg.max_tool_rounds + 1):
            resp = await self._generate()
            content = resp.candidates[0].content if resp.candidates else None
            parts = list(content.parts or []) if content else []
            if content:
                self._contents.append(content)
            spoken += [p.text for p in parts if p.text and not p.thought]
            fcs = [p.function_call for p in parts if p.function_call]
            if not fcs:
                break
            replies = []
            for fc in fcs:
                before = len(self.state.tool_calls)
                result = execute_tool(self.state, fc.name, dict(fc.args or {}))
                calls += self.state.tool_calls[before:]
                replies.append(types.Part.from_function_response(name=fc.name, response=result))
            self._contents.append(types.Content(role="user", parts=replies))
            if self.state.ended or any(fc.name in TERMINAL_TOOLS for fc in fcs):
                break  # call is over; do not ask the model for more
        text = " ".join(s.strip() for s in spoken if s.strip())
        if text:
            self.state.turns.append(Turn("agent", text))
        return TurnResult(text=text, tool_calls=calls, ended=self.state.ended)

    async def _generate(self) -> Any:
        for attempt, delay in enumerate((*_RETRY_DELAYS, None)):
            try:
                return await self._client.aio.models.generate_content(
                    model=self.cfg.agent_model, contents=self._contents, config=self._config
                )
            except Exception as e:  # noqa: BLE001 - only retry rate limits / transient 5xx
                code = getattr(e, "code", None)
                if delay is None or code not in (429, 500, 503):
                    raise
                await asyncio.sleep(delay)


def _user(text: str) -> types.Content:
    return types.Content(role="user", parts=[types.Part(text=text)])

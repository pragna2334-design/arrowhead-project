"""Tool specs (provider-neutral) and handlers. Single source of truth for both modes."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from brain.state import CallState, ToolCall

END_REASONS = ["completed", "declined", "callback_booked", "dnc", "wrong_person", "other"]


@dataclass(frozen=True)
class ToolSpec:
    name: str
    description: str
    properties: dict[str, Any]  # JSON-schema property map
    required: list[str]


TOOL_SPECS: list[ToolSpec] = [
    ToolSpec(
        "record_consent",
        "Record that the customer explicitly agreed to proceed with the loan application "
        "(a clear yes, not silence or 'maybe'). Call only after they say yes.",
        {},
        [],
    ),
    ToolSpec(
        "book_callback",
        "Book a callback at a time the customer chose. Pass it as they said it, "
        "e.g. 'tomorrow 6 pm'.",
        {"time": {"type": "string", "description": "Callback time in the customer's words."}},
        ["time"],
    ),
    ToolSpec(
        "end_call",
        "End the call. Say your goodbye in the same reply BEFORE calling this.",
        {"reason": {"type": "string", "enum": END_REASONS}},
        ["reason"],
    ),
    ToolSpec(
        "mark_do_not_call",
        "Customer asked not to be called again. Call immediately. Say a one-line apology "
        "in the same reply; the call then ends.",
        {},
        [],
    ),
]

TERMINAL_TOOLS = {"end_call", "mark_do_not_call"}


def execute_tool(state: CallState, name: str, args: dict[str, Any] | None) -> dict[str, Any]:
    """Apply a tool call to `state` and return the result sent back to the LLM."""
    args = dict(args or {})
    if name == "record_consent":
        if state.dnc:
            result = {"ok": False, "error": "customer is on do-not-call"}
        else:
            state.consent = True
            result = {"ok": True}
    elif name == "book_callback":
        if state.dnc:
            result = {"ok": False, "error": "customer is on do-not-call"}
        else:
            state.callback_time = str(args.get("time", "")).strip() or None
            result = {"ok": state.callback_time is not None, "time": state.callback_time}
    elif name == "mark_do_not_call":
        state.dnc = True
        state.ended = True
        state.end_reason = "dnc"
        result = {"ok": True}
    elif name == "end_call":
        reason = args.get("reason", "other")
        state.ended = True
        # a DNC end must not be overwritten by a later, vaguer reason
        state.end_reason = "dnc" if state.dnc else (reason if reason in END_REASONS else "other")
        result = {"ok": True}
    else:
        result = {"ok": False, "error": f"unknown tool {name}"}
    state.tool_calls.append(ToolCall(name=name, args=args, result=result))
    return result

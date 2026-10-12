"""Tool specs (provider-neutral), handlers and closing lines. Single source of truth for both modes."""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any

from brain.state import CallState, ToolCall

END_REASONS = ["completed", "declined", "callback_booked", "dnc", "wrong_person", "other"]
_HHMM = re.compile(r"^([01]?\d|2[0-3]):([0-5]\d)$")


@dataclass(frozen=True)
class ToolSpec:
    name: str
    description: str
    properties: dict[str, Any]  # JSON-schema property map
    required: list[str]


TOOL_SPECS: list[ToolSpec] = [
    ToolSpec(
        "record_consent",
        "Record that the customer explicitly agreed to be contacted by an executive to proceed "
        "with the loan application (a clear yes, not silence or 'maybe'). Call only after they "
        "say yes.",
        {},
        [],
    ),
    ToolSpec(
        "book_callback",
        "Book a callback at a day and time the customer chose. Only times inside the allowed "
        "calling window are accepted; on error, offer the nearest allowed time.",
        {
            "day": {
                "type": "string",
                "description": "Day as the customer said it, e.g. 'today', 'tomorrow', 'Saturday'.",
            },
            "time": {"type": "string", "description": "24-hour local time, format HH:MM."},
        },
        ["day", "time"],
    ),
    ToolSpec(
        "end_call",
        "End the call. The system speaks the closing line; do not say goodbye yourself.",
        {"reason": {"type": "string", "enum": END_REASONS}},
        ["reason"],
    ),
    ToolSpec(
        "mark_do_not_call",
        "Customer asked not to be called again. Call immediately. This ends the call and the "
        "system confirms the removal to the customer.",
        {},
        [],
    ),
]

TERMINAL_TOOLS = {"end_call", "mark_do_not_call"}


def parse_hhmm(value: str) -> tuple[int, int] | None:
    m = _HHMM.match(str(value).strip())
    return (int(m.group(1)), int(m.group(2))) if m else None


def in_callback_window(hh: int, mm: int, hours: tuple[int, int]) -> bool:
    start, end = hours
    return start * 60 <= hh * 60 + mm < end * 60


def execute_tool(
    state: CallState,
    name: str,
    args: dict[str, Any] | None,
    *,
    callback_hours: tuple[int, int] = (10, 19),
) -> dict[str, Any]:
    """Apply a tool call to `state` and return the result sent back to the LLM."""
    args = dict(args or {})
    if state.ended:
        result = {"ok": False, "error": "call already ended"}
    elif name == "record_consent":
        if state.dnc:
            result = {"ok": False, "error": "customer is on do-not-call"}
        else:
            state.consent = True
            result = {"ok": True}
    elif name == "book_callback":
        day = str(args.get("day", "")).strip()
        hhmm = parse_hhmm(args.get("time", ""))
        if state.dnc:
            result = {"ok": False, "error": "customer is on do-not-call"}
        elif not day or hhmm is None:
            result = {"ok": False, "error": "need day and time as HH:MM (24-hour)"}
        elif not in_callback_window(*hhmm, callback_hours):
            result = {
                "ok": False,
                "error": f"outside allowed calling hours {callback_hours[0]}:00-"
                f"{callback_hours[1]}:00; offer the nearest allowed time",
            }
        else:
            state.callback_time = f"{day} {hhmm[0]:02d}:{hhmm[1]:02d}"
            result = {"ok": True, "callback_time": state.callback_time}
    elif name == "mark_do_not_call":
        state.dnc = True
        state.ended = True
        state.end_reason = "dnc"
        result = {"ok": True}
    elif name == "end_call":
        reason = args.get("reason", "other")
        state.ended = True
        if state.dnc:
            state.end_reason = "dnc"
        elif reason in END_REASONS:
            state.end_reason = reason
        else:
            state.end_reason = "other"
        result = {"ok": True}
    else:
        result = {"ok": False, "error": f"unknown tool {name}"}
    state.tool_calls.append(
        ToolCall(name=name, args=args, result=result, turn_index=len(state.turns) - 1)
    )
    return result


def closing_line(state: CallState) -> str:
    """Deterministic closing line spoken after a terminal tool, in both modes.

    Deterministic so the DNC confirmation and callback time are always said, and so the
    call never ends in silence if the model calls the tool without speaking.
    """
    if state.dnc or state.end_reason == "dnc":
        return (
            "Bilkul, maaf kijiye. Aapka number hamari calling list se hata diya gaya hai, "
            "ab aapko hamari taraf se call nahi aayega. Dhanyavaad."
        )
    if state.end_reason == "wrong_person":
        return "Maaf kijiye, galat number lag gaya. Aapka din shubh ho."
    if state.callback_time:
        return (
            f"Theek hai, {state.callback_time} par hamare executive aapko call karenge. "
            "Aapke samay ke liye dhanyavaad, aapka din shubh ho!"
        )
    if state.end_reason == "declined":
        return "Koi baat nahi, aapke samay ke liye dhanyavaad. Aapka din shubh ho!"
    return "Aapke samay ke liye dhanyavaad. Aapka din shubh ho!"

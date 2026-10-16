"""Call state shared by voice mode, text mode, simulator and scorer."""

from __future__ import annotations

import time
from dataclasses import asdict, dataclass, field
from typing import Any, Literal

from schemas import CallUsage


@dataclass
class ToolCall:
    name: str
    args: dict[str, Any]
    result: dict[str, Any]
    turn_index: int = -1  # index into CallState.turns of the last turn before the call


@dataclass
class Turn:
    role: Literal["agent", "customer"]
    text: str
    ts: float = field(default_factory=time.time)
    sentiment: str | None = None  # customer turns only: calm | annoyed | confused | interested


@dataclass
class CallState:
    call_id: str
    customer_name: str
    consent: bool | None = None
    callback_time: str | None = None
    dnc: bool = False
    ended: bool = False
    end_reason: str | None = None
    mood: str | None = None  # latest customer mood
    turns: list[Turn] = field(default_factory=list)
    tool_calls: list[ToolCall] = field(default_factory=list)
    usage: CallUsage = field(default_factory=CallUsage)

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        d["usage"] = self.usage.model_dump()
        return d

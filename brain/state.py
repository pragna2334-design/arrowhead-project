"""Call state shared by voice mode, text mode, simulator and scorer."""

from __future__ import annotations

import time
from dataclasses import asdict, dataclass, field
from typing import Any, Literal


@dataclass
class ToolCall:
    name: str
    args: dict[str, Any]
    result: dict[str, Any]


@dataclass
class Turn:
    role: Literal["agent", "customer"]
    text: str
    ts: float = field(default_factory=time.time)


@dataclass
class CallState:
    call_id: str
    customer_name: str
    consent: bool | None = None
    callback_time: str | None = None
    dnc: bool = False
    ended: bool = False
    end_reason: str | None = None
    turns: list[Turn] = field(default_factory=list)
    tool_calls: list[ToolCall] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        """JSON-serialisable form; this is the transcript schema the scorer reads."""
        return asdict(self)

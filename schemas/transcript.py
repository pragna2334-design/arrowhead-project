from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, Field

from schemas.common import EndReason, Mode, Mood


class TranscriptTurn(BaseModel):
    role: Literal["agent", "customer"]
    text: str  # PII-redacted, without the [customer_mood] tag
    ts: float
    sentiment: Mood | None = None  # customer turns only


class TranscriptToolCall(BaseModel):
    name: str
    args: dict[str, Any]
    result: dict[str, Any]
    turn_index: int  # index of the last turn before the call (-1 = before any turn)


class FinalState(BaseModel):
    consent: bool | None = None
    callback_time: str | None = None  # "<day> HH:MM"
    dnc: bool = False
    ended: bool = False
    end_reason: EndReason | None = None


class Transcript(BaseModel):
    """One call. Written to results/<version>/transcripts/<call_id>.json (text mode) or
    logs/calls/<call_id>.transcript.json (voice mode)."""

    schema_version: int = 1
    call_id: str
    mode: Mode
    prompt_version: str
    agent_model: str
    persona_id: str | None = None  # None for voice calls
    customer_name: str
    started_at: float
    ended_at: float
    stop_reason: Literal["agent_ended", "max_turns", "customer_hung_up", "error"]
    error: str | None = None
    turns: list[TranscriptTurn] = Field(default_factory=list)
    tool_calls: list[TranscriptToolCall] = Field(default_factory=list)
    final_state: FinalState

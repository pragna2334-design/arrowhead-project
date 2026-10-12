"""Convert a finished CallState into the frozen Transcript schema (PII-redacted)."""

from __future__ import annotations

import time

from brain.redact import redact, redact_obj
from brain.state import CallState
from schemas import FinalState, Transcript, TranscriptToolCall, TranscriptTurn


def to_transcript(
    state: CallState,
    *,
    mode: str,
    prompt_version: str,
    agent_model: str,
    started_at: float,
    stop_reason: str,
    persona_id: str | None = None,
    ended_at: float | None = None,
    error: str | None = None,
) -> Transcript:
    return Transcript(
        call_id=state.call_id,
        mode=mode,
        prompt_version=prompt_version,
        agent_model=agent_model,
        persona_id=persona_id,
        customer_name=state.customer_name,
        started_at=started_at,
        ended_at=ended_at or time.time(),
        stop_reason=stop_reason,
        error=redact(error) if error else None,
        turns=[
            TranscriptTurn(role=t.role, text=redact(t.text), ts=t.ts, sentiment=t.sentiment)
            for t in state.turns
        ],
        tool_calls=[
            TranscriptToolCall(
                name=c.name,
                args=redact_obj(c.args),
                result=redact_obj(c.result),
                turn_index=c.turn_index,
            )
            for c in state.tool_calls
        ],
        final_state=FinalState(
            consent=state.consent,
            callback_time=state.callback_time,
            dnc=state.dnc,
            ended=state.ended,
            end_reason=state.end_reason,
        ),
    )

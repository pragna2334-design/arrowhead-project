from __future__ import annotations

from typing import Literal

from pydantic import BaseModel


class LatencyRecord(BaseModel):
    """One line of logs/latency.jsonl (one per bot turn in voice mode). Times in ms."""

    call_id: str
    prompt_version: str | None = None
    turn: int
    ts: float
    kind: Literal["greeting", "turn"]  # greeting is timed from client connect; exclude from p50/p95
    stt_ms: float | None = None  # VAD stop -> final transcript
    llm_ttfb_ms: float | None = None
    tts_ttfb_ms: float | None = None
    pipeline_ms: float | None = None  # STT final -> first TTS audio
    e2e_ms: float | None = None  # user silence -> bot starts speaking
    contributions_ms: dict[str, float | None] = {}

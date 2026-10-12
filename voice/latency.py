"""Per-turn latency logging to JSONL, fed by Pipecat's UserBotLatencyObserver.

Two end-to-end definitions are logged for every bot turn (both in ms):
  e2e_ms       user silence -> bot starts speaking (what the caller feels; includes the
               VAD endpointing wait, STT, turn detection, LLM, TTS and output transport).
  pipeline_ms  STT final transcript -> first TTS audio (our pipeline only: turn detection,
               LLM inference, tool/handler time, sentence aggregation, TTS synthesis).
Per-stage fields: stt_ms (VAD stop -> final transcript), llm_ttfb_ms, tts_ttfb_ms.
"""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any

from pipecat.observers.user_bot_latency_observer import LatencyBreakdown, MeasuredFrom

from schemas import LatencyRecord

# Contribution keys that are NOT part of "STT final -> first TTS audio".
_OUTSIDE_PIPELINE = {"endpointing_wait", "transcription", "first_request", "output_transport"}


def _ms(secs: float | None) -> float | None:
    return None if secs is None else round(secs * 1000, 1)


def breakdown_to_record(
    b: LatencyBreakdown, call_id: str, turn: int, prompt_version: str | None = None
) -> dict[str, Any]:
    by_key: dict[str, float] = {}
    for c in b.contributions:
        by_key[c.key] = by_key.get(c.key, 0.0) + c.duration_secs

    def ttfb(suffix: str) -> float | None:
        # Match the service class name ("SarvamTTSService#0" -> "SarvamTTSService") by suffix:
        # a substring test for "TTS" would also hit "DeepgramSTTService" (S-TTS-ervice).
        vals = [
            t.duration_secs for t in b.ttfb if t.processor.split("#")[0].upper().endswith(suffix)
        ]
        return max(vals) if vals else None

    pipeline = sum(v for k, v in by_key.items() if k not in _OUTSIDE_PIPELINE)
    return {
        "call_id": call_id,
        "prompt_version": prompt_version,
        "turn": turn,
        "ts": time.time(),
        "kind": "greeting" if b.measured_from is MeasuredFrom.CLIENT_CONNECTED else "turn",
        "stt_ms": _ms(by_key.get("transcription")),
        "llm_ttfb_ms": _ms(ttfb("LLMSERVICE")),
        "tts_ttfb_ms": _ms(ttfb("TTSSERVICE")),
        "pipeline_ms": _ms(pipeline) if by_key else None,
        "e2e_ms": _ms(b.total_secs),
        "contributions_ms": {k: _ms(v) for k, v in by_key.items()},
    }


def attach_latency_logging(
    observer: Any, call_id: str, path: str | Path, prompt_version: str | None = None
) -> None:
    """Register an on_latency_breakdown handler that appends one JSON line per bot turn."""
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    counter = {"turn": 0}

    @observer.event_handler("on_latency_breakdown")
    async def _on_breakdown(_observer, breakdown: LatencyBreakdown):
        counter["turn"] += 1
        record = breakdown_to_record(breakdown, call_id, counter["turn"], prompt_version)
        LatencyRecord.model_validate(record)  # frozen contract
        with out.open("a") as f:  # noqa: ASYNC230 - one tiny append per turn
            f.write(json.dumps(record, ensure_ascii=False) + "\n")

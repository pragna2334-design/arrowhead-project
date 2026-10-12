import asyncio
import json

from pipecat.observers.user_bot_latency_observer import (
    LatencyBreakdown,
    LatencyContribution,
    LatencyOwnerKind,
    MeasuredFrom,
    TTFBBreakdownMetrics,
    UserBotLatencyObserver,
)

from schemas import LatencyRecord
from voice.latency import attach_latency_logging, breakdown_to_record


def contrib(key, secs):
    return LatencyContribution(
        key=key,
        label=key,
        owner="x",
        owner_kind=LatencyOwnerKind.SERVICE,
        start_time=0,
        duration_secs=secs,
    )


def breakdown(measured_from=MeasuredFrom.USER_SILENCE):
    return LatencyBreakdown(
        ttfb=[
            TTFBBreakdownMetrics(
                processor="DeepSeekLLMService#0", start_time=0, duration_secs=0.31
            ),
            TTFBBreakdownMetrics(processor="SarvamTTSService#0", start_time=0, duration_secs=0.22),
            TTFBBreakdownMetrics(processor="DeepgramSTTService#0", start_time=0, duration_secs=0.1),
        ],
        contributions=[
            contrib("endpointing_wait", 0.2),
            contrib("transcription", 0.15),
            contrib("turn_detection", 0.01),
            contrib("llm_inference", 0.31),
            contrib("speech_synthesis", 0.22),
            contrib("output_transport", 0.03),
        ],
        measured_from=measured_from,
        total_secs=0.92,
    )


def test_record_definitions():
    r = breakdown_to_record(breakdown(), "c", 2, "v1")
    LatencyRecord.model_validate(r)
    assert r["e2e_ms"] == 920.0 and r["pipeline_ms"] == 540.0 and r["stt_ms"] == 150.0
    assert r["llm_ttfb_ms"] == 310.0 and r["tts_ttfb_ms"] == 220.0
    assert r["kind"] == "turn" and r["prompt_version"] == "v1"


def test_greeting_kind():
    r = breakdown_to_record(breakdown(MeasuredFrom.CLIENT_CONNECTED), "c", 1)
    assert r["kind"] == "greeting"


def test_observer_writes_jsonl(tmp_path):
    obs = UserBotLatencyObserver()
    path = tmp_path / "l.jsonl"
    attach_latency_logging(obs, "c", path, "v2")
    asyncio.run(obs._call_event_handler("on_latency_breakdown", breakdown()))
    rec = json.loads(path.read_text().splitlines()[0])
    assert rec["turn"] == 1 and rec["prompt_version"] == "v2"

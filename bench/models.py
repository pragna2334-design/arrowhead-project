"""JSON shape of one bench run (logs/bench/<label>-<timestamp>.json). Times in ms."""

from __future__ import annotations

from pydantic import BaseModel, Field

# Metrics summarised as p50/p95, in table order.
METRICS = (
    "llm_ttfb_ms",
    "llm_first_text_ms",
    "llm_first_clause_ms",
    "llm_total_ms",
    "tts_ttfb_ms",
    "est_first_audio_ms",
    "reasoning_chunks",
)


class TTSMeta(BaseModel):
    provider: str = "sarvam"
    transport: str = "websocket"  # sarvamai SDK streaming socket (same endpoint as Pipecat)
    model: str
    voice: str
    language: str
    min_buffer_size: int
    sample_rate: int


class BenchMeta(BaseModel):
    label: str
    created_at: str  # ISO-8601 UTC
    model: str
    thinking: str  # enabled | disabled
    temperature: float
    max_tokens: int
    prompt_version: str
    scenario: str
    aggregation: str  # sentence | clause | token: how llm_first_clause_ms was cut
    utterances: int
    repeats: int
    tts: TTSMeta | None = None
    tts_skipped_reason: str | None = None


class BenchSample(BaseModel):
    utterance_id: str
    repeat: int
    mood: str | None = None
    llm_ttfb_ms: float | None = None  # request sent -> first streamed chunk (any kind)
    llm_first_text_ms: float | None = None  # -> first content token
    llm_first_clause_ms: float | None = None  # -> first chunk the aggregator releases to TTS
    llm_total_ms: float | None = None  # -> stream finished
    reasoning_chunks: int = 0  # chunks with reasoning_content: >0 means thinking is not off
    tool_call: bool = False
    tool_names: list[str] = Field(default_factory=list)
    first_clause: str | None = None
    first_clause_chars: int | None = None
    reply_text: str = ""
    tts_connect_ms: float | None = None  # websocket connect + config (not in TTFB)
    tts_ttfb_ms: float | None = None  # first clause sent -> first audio byte
    est_first_audio_ms: float | None = None  # llm_first_clause_ms + tts_ttfb_ms
    error: str | None = None


class MetricStats(BaseModel):
    n: int
    p50: float | None = None
    p95: float | None = None
    mean: float | None = None
    min: float | None = None
    max: float | None = None


class BenchReport(BaseModel):
    schema_version: int = 1
    meta: BenchMeta
    samples: list[BenchSample]
    stats: dict[str, MetricStats]

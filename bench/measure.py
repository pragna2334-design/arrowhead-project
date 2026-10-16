"""Measurement primitives: LLM stream timing, Sarvam TTS first-audio timing, pacing, retries.

Every network client, the clock and sleep are injected, so all of this runs offline in tests.
"""

from __future__ import annotations

import asyncio
import time
from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field
from typing import Any, Protocol, TypeVar

from voice.chunking import ClauseChunker, parse_aggregation_mode

Clock = Callable[[], float]
Sleep = Callable[[float], Awaitable[Any]]
T = TypeVar("T")

RETRY_DELAYS = (1.0, 2.0, 4.0, 8.0, 16.0)
_TRANSIENT_STATUS = {429, 500, 502, 503}


def _ms(secs: float) -> float:
    return round(secs * 1000, 1)


# --- text splitters: what the voice pipeline would release to TTS ----------------------------


class Splitter(Protocol):
    async def feed(self, text: str) -> list[str]: ...
    async def flush(self) -> str | None: ...


class ClauseSplitter:
    def __init__(self):
        self._c = ClauseChunker()

    async def feed(self, text: str) -> list[str]:
        return self._c.feed(text)

    async def flush(self) -> str | None:
        return self._c.flush()


class TokenSplitter:
    async def feed(self, text: str) -> list[str]:
        return [text] if text.strip() else []

    async def flush(self) -> str | None:
        return None


class SentenceSplitter:
    """Pipecat's own SimpleTextAggregator in SENTENCE mode (what the bot used before)."""

    def __init__(self, language: str | None = None):
        from pipecat.utils.text.base_text_aggregator import AggregationType
        from pipecat.utils.text.simple_text_aggregator import SimpleTextAggregator

        self._agg = SimpleTextAggregator(
            aggregation_type=AggregationType.SENTENCE, language=language
        )

    async def feed(self, text: str) -> list[str]:
        return [a.text async for a in self._agg.aggregate(text) if a.text.strip()]

    async def flush(self) -> str | None:
        rest = await self._agg.flush()
        return rest.text if rest and rest.text.strip() else None


def make_splitter(mode: str, language: str | None = None) -> Splitter:
    mode = parse_aggregation_mode(mode)
    if mode == "clause":
        return ClauseSplitter()
    if mode == "token":
        return TokenSplitter()
    return SentenceSplitter(language)


# --- LLM ---------------------------------------------------------------------------------------


@dataclass
class LLMMeasurement:
    ttfb_ms: float | None = None
    first_text_ms: float | None = None
    first_clause_ms: float | None = None
    total_ms: float | None = None
    reasoning_chunks: int = 0
    tool_names: list[str] = field(default_factory=list)
    first_clause: str | None = None
    text: str = ""

    @property
    def tool_call(self) -> bool:
        return bool(self.tool_names)


async def measure_llm(
    client: Any,
    *,
    model: str,
    messages: list[dict[str, Any]],
    tools: list[dict[str, Any]] | None,
    temperature: float,
    max_tokens: int,
    thinking: str,
    splitter: Splitter,
    clock: Clock = time.perf_counter,
) -> LLMMeasurement:
    """Stream one chat completion with the agent's exact settings and time it."""
    m = LLMMeasurement()
    t0 = clock()
    stream = await client.chat.completions.create(
        model=model,
        messages=messages,
        tools=tools or None,
        temperature=temperature,
        max_tokens=max_tokens,
        stream=True,
        stream_options={"include_usage": True},  # as Pipecat's DeepSeekLLMService sends it
        extra_body={"thinking": {"type": thinking}},  # same field Pipecat and the brain send
    )
    parts: list[str] = []
    async for chunk in stream:
        now = clock()
        if m.ttfb_ms is None:
            m.ttfb_ms = _ms(now - t0)
        choices = getattr(chunk, "choices", None) or []
        if not choices:
            continue  # usage-only chunk
        delta = getattr(choices[0], "delta", None)
        if delta is None:
            continue
        if getattr(delta, "reasoning_content", None):
            m.reasoning_chunks += 1
        for tc in getattr(delta, "tool_calls", None) or []:
            name = getattr(getattr(tc, "function", None), "name", None)
            if name:
                m.tool_names.append(name)
        content = getattr(delta, "content", None)
        if content:
            if m.first_text_ms is None:
                m.first_text_ms = _ms(now - t0)
            parts.append(content)
            released = await splitter.feed(content)
            if released and m.first_clause is None:
                m.first_clause, m.first_clause_ms = released[0], _ms(now - t0)
    end = clock()
    m.total_ms = _ms(end - t0)
    rest = await splitter.flush()
    if m.first_clause is None and rest:
        m.first_clause, m.first_clause_ms = rest, m.total_ms  # released at end of response
    m.text = "".join(parts).strip()
    return m


# --- TTS ---------------------------------------------------------------------------------------


@dataclass
class TTSMeasurement:
    ttfb_ms: float
    connect_ms: float


class TTSProbe(Protocol):
    async def first_audio(self, text: str) -> TTSMeasurement: ...


class SarvamStreamingProbe:
    """Time to first audio byte over Sarvam's TTS websocket (sarvamai SDK streaming client).

    Same endpoint and config fields Pipecat's SarvamTTSService sends. A fresh socket per sample;
    connect + config time is reported separately (Pipecat keeps one socket open per call).
    The clause is flushed right away, i.e. synthesized even if shorter than min_buffer_size.
    """

    def __init__(
        self,
        *,
        api_key: str,
        model: str,
        voice: str,
        language: str,
        min_buffer_size: int = 50,
        max_chunk_length: int = 150,
        sample_rate: int = 24000,
        client: Any = None,
        clock: Clock = time.perf_counter,
    ):
        self.model, self.voice, self.language = model, voice, language
        self.min_buffer_size, self.max_chunk_length = min_buffer_size, max_chunk_length
        self.sample_rate = sample_rate
        self._clock = clock
        if client is None:
            from sarvamai import AsyncSarvamAI

            client = AsyncSarvamAI(api_subscription_key=api_key)
        self._client = client

    def _config_message(self) -> Any:
        from sarvamai.types.configure_connection import ConfigureConnection
        from sarvamai.types.configure_connection_data import ConfigureConnectionData

        # Exactly Pipecat's _send_config fields for bulbul:v3 (no pitch/loudness, which the
        # SDK's configure() would always add).
        return ConfigureConnection(
            data=ConfigureConnectionData(
                target_language_code=self.language,
                speaker=self.voice,
                speech_sample_rate=self.sample_rate,
                enable_preprocessing=True,  # always on for bulbul:v3
                min_buffer_size=self.min_buffer_size,
                max_chunk_length=self.max_chunk_length,
                output_audio_codec="linear16",
                output_audio_bitrate="128k",
                pace=1.0,
                model=self.model,
            )
        )

    async def first_audio(self, text: str) -> TTSMeasurement:
        clock = self._clock
        t0 = clock()
        async with self._client.text_to_speech_streaming.connect(
            model=self.model, send_completion_event=True
        ) as ws:
            await ws._send_model(self._config_message())
            t1 = clock()
            await ws.convert(text)
            await ws.flush()
            async for msg in ws:
                kind = getattr(msg, "type", None)
                if kind == "audio":
                    return TTSMeasurement(ttfb_ms=_ms(clock() - t1), connect_ms=_ms(t1 - t0))
                if kind == "error":
                    data = getattr(msg, "data", None)
                    raise RuntimeError(f"Sarvam TTS error: {getattr(data, 'message', data)}")
        raise RuntimeError("Sarvam TTS closed without audio")


# --- pacing and retries ------------------------------------------------------------------------


class Pacer:
    """At most one request start per `interval` seconds."""

    def __init__(
        self, interval: float, *, clock: Clock = time.monotonic, sleep: Sleep = asyncio.sleep
    ):
        self.interval, self._clock, self._sleep = interval, clock, sleep
        self._last: float | None = None

    async def wait(self) -> None:
        if self._last is not None:
            gap = self.interval - (self._clock() - self._last)
            if gap > 0:
                await self._sleep(gap)
        self._last = self._clock()


def is_transient(e: BaseException) -> bool:
    if getattr(e, "status_code", None) in _TRANSIENT_STATUS:
        return True
    try:
        from openai import APIConnectionError
    except ImportError:  # pragma: no cover
        return False
    return isinstance(e, APIConnectionError)


async def with_retries(
    fn: Callable[[], Awaitable[T]],
    *,
    sleep: Sleep = asyncio.sleep,
    delays: tuple[float, ...] = RETRY_DELAYS,
    on_retry: Callable[[BaseException, float], None] | None = None,
) -> T:
    """Call fn; on 429/5xx/connection errors back off and retry (len(delays) retries)."""
    for delay in (*delays, None):
        try:
            return await fn()
        except Exception as e:
            if delay is None or not is_transient(e):
                raise
            if on_retry:
                on_retry(e, delay)
            await sleep(delay)
    raise AssertionError("unreachable")

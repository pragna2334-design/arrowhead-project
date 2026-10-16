"""Pipecat glue for `tts.<provider>.text_aggregation` (sentence | clause | token).

How it hooks into Pipecat 1.12.0 (verified against the installed source):

- sentence: TTSService(text_aggregation_mode=TextAggregationMode.SENTENCE), Pipecat's default
  SimpleTextAggregator.
- token: TTSService(text_aggregation_mode=TextAggregationMode.TOKEN); every LLM token goes
  straight to the TTS service.
- clause: the service is built in SENTENCE mode, then its text aggregator is replaced with
  ClauseTextAggregator (a BaseTextAggregator). TTSService keeps the aggregator in
  `self._text_aggregator` and only calls aggregate(), flush(), handle_interruption(),
  set_language() and .language on it; Pipecat's own Cartesia and Rime services replace it the
  same way. Doing it inside the TTS service (rather than with an LLMTextProcessor in front of
  it) keeps the TextAggregationMetricsData the latency observer turns into its
  "sentence_aggregation" contribution, now measuring the time to the first clause.
"""

from __future__ import annotations

from collections.abc import AsyncIterator
from typing import Any

from pipecat.services.tts_service import TextAggregationMode
from pipecat.utils.text.base_text_aggregator import (
    Aggregation,
    AggregationType,
    BaseTextAggregator,
)

from voice.chunking import FIRST_MIN_CHARS, MIN_CHARS, ClauseChunker, parse_aggregation_mode


class ClauseTextAggregator(BaseTextAggregator):
    """BaseTextAggregator that releases text at clause boundaries (see voice.chunking)."""

    def __init__(self, *, first_min_chars: int = FIRST_MIN_CHARS, min_chars: int = MIN_CHARS):
        # Report clauses as SENTENCE aggregations: everything downstream (TTS metrics, the
        # aggregated-frame sequencer, RTVI) then treats them exactly like sentences.
        super().__init__(aggregation_type=AggregationType.SENTENCE)
        self._chunker = ClauseChunker(first_min_chars=first_min_chars, min_chars=min_chars)

    @property
    def text(self) -> Aggregation:
        return Aggregation(text=self._chunker.pending_text, type=AggregationType.SENTENCE)

    async def aggregate(self, text: str) -> AsyncIterator[Aggregation]:
        for clause in self._chunker.feed(text):
            yield Aggregation(text=clause, type=AggregationType.SENTENCE)

    async def flush(self) -> Aggregation | None:
        rest = self._chunker.flush()
        return Aggregation(text=rest, type=AggregationType.SENTENCE) if rest else None

    async def handle_interruption(self):
        self._chunker.reset()

    async def reset(self):
        self._chunker.reset()


def tts_aggregation_kwargs(mode: str) -> dict[str, Any]:
    """Constructor kwargs for a Pipecat TTS service. Clause mode is built in SENTENCE mode."""
    mode = parse_aggregation_mode(mode)
    tam = TextAggregationMode.TOKEN if mode == "token" else TextAggregationMode.SENTENCE
    return {"text_aggregation_mode": tam}


def install_clause_aggregator(tts: Any, aggregator: BaseTextAggregator | None = None) -> Any:
    """Swap the TTS service's sentence aggregator for a ClauseTextAggregator. Returns tts.

    Fails loudly at startup if a Pipecat upgrade changes the internals this relies on.
    """
    current = getattr(tts, "_text_aggregator", None)
    if not isinstance(current, BaseTextAggregator):
        raise TypeError(
            f"{type(tts).__name__} has no BaseTextAggregator at _text_aggregator; "
            "clause aggregation needs re-checking against this Pipecat version"
        )
    if getattr(tts, "_text_aggregation_mode", None) != TextAggregationMode.SENTENCE:
        raise RuntimeError("clause aggregation requires the TTS service in SENTENCE mode")
    new = aggregator or ClauseTextAggregator()
    new.set_language(current.language)
    tts._text_aggregator = new
    return tts


def apply_text_aggregation(tts: Any, mode: str) -> Any:
    """Post-construction step: install the clause aggregator when mode == 'clause'."""
    if parse_aggregation_mode(mode) == "clause":
        install_clause_aggregator(tts)
    return tts

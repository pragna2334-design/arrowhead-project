"""Pipecat voice pipeline: browser mic (SmallWebRTC) -> STT -> DeepSeek (brain) -> TTS -> browser.

Run:  python -m voice.bot -t webrtc      then open http://localhost:7860/client
All prompt / tool / state logic comes from the shared `brain` package.
"""

from __future__ import annotations

import os
import time
import uuid

from dotenv import load_dotenv
from loguru import logger
from pipecat.adapters.schemas.function_schema import FunctionSchema
from pipecat.adapters.schemas.tools_schema import ToolsSchema
from pipecat.audio.vad.silero import SileroVADAnalyzer
from pipecat.audio.vad.vad_analyzer import VADParams
from pipecat.frames.frames import (
    EndWorkerFrame,
    FunctionCallResultProperties,
    LLMRunFrame,
    TTSSpeakFrame,
)
from pipecat.observers.user_bot_latency_observer import UserBotLatencyObserver
from pipecat.pipeline.pipeline import Pipeline
from pipecat.pipeline.worker import PipelineParams, PipelineWorker
from pipecat.processors.aggregators.llm_context import LLMContext
from pipecat.processors.aggregators.llm_response_universal import (
    LLMContextAggregatorPair,
    LLMUserAggregatorParams,
)
from pipecat.runner.types import RunnerArguments
from pipecat.runner.utils import create_transport
from pipecat.services.deepgram.stt import DeepgramSTTService
from pipecat.services.deepgram.tts import DeepgramTTSService
from pipecat.services.deepseek.llm import DeepSeekLLMService
from pipecat.services.elevenlabs.tts import ElevenLabsTTSService
from pipecat.services.llm_service import FunctionCallParams
from pipecat.services.sarvam.stt import SarvamSTTService
from pipecat.services.sarvam.tts import SarvamTTSService
from pipecat.transcriptions.language import Language
from pipecat.transports.base_transport import BaseTransport, TransportParams
from pipecat.turns.user_mute.function_call_user_mute_strategy import FunctionCallUserMuteStrategy
from pipecat.turns.user_mute.mute_until_first_bot_complete_user_mute_strategy import (
    MuteUntilFirstBotCompleteUserMuteStrategy,
)
from pipecat.turns.user_start.min_words_user_turn_start_strategy import (
    MinWordsUserTurnStartStrategy,
)
from pipecat.turns.user_turn_strategies import UserTurnStrategies
from pipecat.workers.runner import WorkerRunner

from brain.config import AppConfig, load_config
from brain.postcall import write_voice_records
from brain.prompt import OPENING_CUE, build_system_prompt
from brain.redact import install_log_redaction
from brain.state import CallState
from brain.tools import TERMINAL_TOOLS, closing_line, execute_tool, tool_specs
from brain.transcript import to_transcript
from voice.aggregation import apply_text_aggregation, tts_aggregation_kwargs
from voice.chunking import DEFAULT_AGGREGATION, parse_aggregation_mode
from voice.context import context_to_turns
from voice.latency import attach_latency_logging
from voice.llm_settings import log_thinking, validate_thinking
from voice.mood import MoodTagger

load_dotenv(override=True)
install_log_redaction()  # every log line passes through PII redaction

# Deferred so only the selected transport's params are built. Browser/WebRTC only.
transport_params = {
    "webrtc": lambda: TransportParams(audio_in_enabled=True, audio_out_enabled=True),
}


def build_stt(cfg: AppConfig):
    provider = cfg.stt["provider"]
    if provider == "deepgram":
        c = cfg.stt["deepgram"]
        return DeepgramSTTService(
            api_key=os.environ["DEEPGRAM_API_KEY"],
            settings=DeepgramSTTService.Settings(model=c["model"], language=c["language"]),
        )
    if provider == "sarvam":
        return SarvamSTTService(
            api_key=os.environ["SARVAM_API_KEY"], mode=cfg.stt.get("sarvam", {}).get("mode")
        )
    raise ValueError(f"unknown stt.provider: {provider}")


# Default chunking per TTS provider when its config has no `text_aggregation` key.
DEFAULT_TEXT_AGGREGATION = {
    "sarvam": DEFAULT_AGGREGATION,
    "deepgram": "sentence",
    "elevenlabs": "sentence",
}


def text_aggregation_mode(cfg: AppConfig) -> str:
    """tts.<provider>.text_aggregation (sentence | clause | token), validated."""
    provider = cfg.tts["provider"]
    section = cfg.tts.get(provider) or {}
    return parse_aggregation_mode(
        section.get("text_aggregation", DEFAULT_TEXT_AGGREGATION.get(provider, "sentence"))
    )


def build_tts(cfg: AppConfig):
    provider = cfg.tts["provider"]
    if provider not in DEFAULT_TEXT_AGGREGATION:
        raise ValueError(f"unknown tts.provider: {provider}")
    mode = text_aggregation_mode(cfg)
    agg = tts_aggregation_kwargs(mode)
    if provider == "sarvam":
        c = cfg.tts["sarvam"]
        extra = {}
        if c.get("min_buffer_size") is not None:
            # Sarvam's websocket holds text until this many chars arrive (Pipecat default 50);
            # a short first clause is only synthesized once the buffer fills or the turn ends.
            extra["min_buffer_size"] = int(c["min_buffer_size"])
        tts = SarvamTTSService(
            api_key=os.environ["SARVAM_API_KEY"],
            settings=SarvamTTSService.Settings(
                model=c["model"], voice=c["voice"], language=Language(c["language"]), **extra
            ),
            **agg,
        )
    elif provider == "deepgram":
        tts = DeepgramTTSService(
            api_key=os.environ["DEEPGRAM_API_KEY"],
            settings=DeepgramTTSService.Settings(voice=cfg.tts["deepgram"]["voice"]),
            **agg,
        )
    else:
        c = cfg.tts["elevenlabs"]
        tts = ElevenLabsTTSService(
            api_key=os.environ["ELEVENLABS_API_KEY"],
            settings=ElevenLabsTTSService.Settings(voice=c["voice_id"], model=c["model"]),
            **agg,
        )
    logger.info(f"TTS {provider}: text_aggregation={mode}")
    return apply_text_aggregation(tts, mode)


def build_llm(cfg: AppConfig, customer_name: str):
    b = cfg.brain
    thinking = validate_thinking(b.thinking)  # ValueError on anything but enabled|disabled
    log_thinking(b.agent_model, thinking)
    return DeepSeekLLMService(
        api_key=os.environ["DEEPSEEK_API_KEY"],
        settings=DeepSeekLLMService.Settings(
            model=b.agent_model,  # pinned in config.yaml, never Pipecat's default
            system_instruction=build_system_prompt(b, customer_name),
            temperature=b.temperature,
            max_tokens=b.max_output_tokens,
            thinking=DeepSeekLLMService.ThinkingConfig(type=thinking),
        ),
    )


def make_tools_schema(cfg: AppConfig) -> ToolsSchema:
    return ToolsSchema(
        standard_tools=[
            FunctionSchema(
                name=s.name, description=s.description, properties=s.properties, required=s.required
            )
            for s in tool_specs(cfg.scenario)
        ]
    )


def make_tool_handler(state: CallState, name: str, cfg: AppConfig):
    async def handler(params: FunctionCallParams):
        was_ended = state.ended
        result = execute_tool(
            state,
            name,
            params.arguments,
            callback_hours=cfg.brain.callback_hours,
            scenario=cfg.scenario,
        )
        if state.tool_calls:  # turn index in the spoken conversation (for the transcript)
            state.tool_calls[-1].turn_index = (
                len(context_to_turns(params.context.get_messages())) - 1
            )
        if name in TERMINAL_TOOLS and state.ended and not was_ended:
            # Deterministic closing line (same as text mode), then a graceful hang-up.
            # Ordering: TTSSpeakFrame and EndWorkerFrame both travel downstream from the LLM;
            # the worker turns EndWorkerFrame into an EndFrame queued behind the speech, and
            # the output transport drains queued audio before stopping.
            await params.result_callback(
                result, properties=FunctionCallResultProperties(run_llm=False)
            )
            await params.llm.push_frame(TTSSpeakFrame(closing_line(state)))
            await params.llm.push_frame(EndWorkerFrame(reason=state.end_reason))
        elif name in TERMINAL_TOOLS:
            # Duplicate end request in the same response: acknowledge, never speak twice.
            await params.result_callback(
                result, properties=FunctionCallResultProperties(run_llm=False)
            )
        else:
            await params.result_callback(result)

    return handler


def build_user_params(cfg: AppConfig) -> LLMUserAggregatorParams:
    """VAD + barge-in tuning.

    - Barge-in: while the bot speaks, the customer must say >= interrupt_min_words words
      (interim STT) to interrupt, so "haan", "hmm" or a cough do not cut the agent off.
      When the bot is silent one word starts a turn. interrupt_min_words: 0 -> Pipecat
      defaults (VAD-triggered interruption).
    - Mute the customer until the first bot utterance completes (the AI + lender disclosure
      is always heard in full) and during tool calls (so a barge-in cannot cancel
      record_consent / book_callback / end_call mid-flight).
    """
    min_words = int(cfg.voice.get("interrupt_min_words", 2))
    strategies = (
        UserTurnStrategies(start=[MinWordsUserTurnStartStrategy(min_words=min_words)])
        if min_words > 0
        else UserTurnStrategies()
    )
    return LLMUserAggregatorParams(
        vad_analyzer=SileroVADAnalyzer(
            params=VADParams(
                start_secs=float(cfg.voice.get("vad_start_secs", 0.2)),
                stop_secs=float(cfg.voice["vad_stop_secs"]),
            )
        ),
        user_turn_strategies=strategies,
        user_mute_strategies=[
            MuteUntilFirstBotCompleteUserMuteStrategy(),
            FunctionCallUserMuteStrategy(),
        ],
    )


async def run_bot(transport: BaseTransport, runner_args: RunnerArguments):
    cfg = load_config()
    customer = cfg.brain.default_customer_name
    state = CallState(call_id=f"voice-{uuid.uuid4().hex[:8]}", customer_name=customer)
    started_at = time.time()
    logger.info(f"Starting call {state.call_id} (prompt {cfg.brain.prompt_version})")

    stt, tts, llm = build_stt(cfg), build_tts(cfg), build_llm(cfg, customer)
    for spec in tool_specs(cfg.scenario):
        llm.register_function(spec.name, make_tool_handler(state, spec.name, cfg))

    context = LLMContext(tools=make_tools_schema(cfg))
    user_aggregator, assistant_aggregator = LLMContextAggregatorPair(
        context, user_params=build_user_params(cfg)
    )

    pipeline = Pipeline(
        [
            transport.input(),
            stt,
            user_aggregator,
            MoodTagger(state),  # same mood tag as text mode, before the LLM sees the turn
            llm,
            tts,
            transport.output(),
            assistant_aggregator,
        ]
    )

    latency_observer = UserBotLatencyObserver()
    attach_latency_logging(
        latency_observer, state.call_id, cfg.voice["latency_log"], cfg.brain.prompt_version
    )

    worker = PipelineWorker(
        pipeline,
        params=PipelineParams(enable_metrics=True, enable_usage_metrics=True),
        observers=[latency_observer],
    )
    runner = WorkerRunner(handle_sigint=runner_args.handle_sigint)
    await runner.add_workers(worker)

    @transport.event_handler("on_client_connected")
    async def on_client_connected(transport, client):
        logger.info("Client connected")
        # Same opening cue the text-mode agent uses, so the agent speaks first.
        context.add_message({"role": "user", "content": OPENING_CUE})
        await worker.queue_frames([LLMRunFrame()])

    @transport.event_handler("on_client_disconnected")
    async def on_client_disconnected(transport, client):
        logger.info("Client disconnected")
        await runner.cancel()

    try:
        await runner.run()
    finally:
        write_postcall(state, context, cfg, started_at)


def write_postcall(state: CallState, context: LLMContext, cfg: AppConfig, started_at: float):
    """Persist the redacted transcript + CRM record for this voice call (never raises)."""
    try:
        state.turns = context_to_turns(context.get_messages(), started_at=started_at)
        transcript = to_transcript(
            state,
            mode="voice",
            prompt_version=cfg.brain.prompt_version,
            agent_model=cfg.brain.agent_model,
            started_at=started_at,
            stop_reason="agent_ended" if state.ended else "customer_hung_up",
            scenario=cfg.scenario.id,
        )
        paths = write_voice_records(transcript, cfg.voice.get("calls_dir", "logs/calls"))
        logger.info(f"Post-call records: {', '.join(str(p) for p in paths)}")
    except Exception as e:  # noqa: BLE001 - a failed write must not crash the runner
        logger.error(f"Post-call record failed for {state.call_id}: {e}")


async def bot(runner_args: RunnerArguments):
    """Entry point discovered by pipecat.runner."""
    transport = await create_transport(runner_args, transport_params)
    await run_bot(transport, runner_args)


if __name__ == "__main__":
    from pipecat.runner.run import main

    main()

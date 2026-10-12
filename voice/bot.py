"""Pipecat voice pipeline: browser mic (SmallWebRTC) -> STT -> Gemini (brain) -> TTS -> browser.

Run:  python -m voice.bot -t webrtc      then open http://localhost:7860/client
All prompt / tool / state logic comes from the shared `brain` package.
"""

from __future__ import annotations

import os
import uuid

from dotenv import load_dotenv
from loguru import logger
from pipecat.adapters.schemas.function_schema import FunctionSchema
from pipecat.adapters.schemas.tools_schema import ToolsSchema
from pipecat.audio.vad.silero import SileroVADAnalyzer
from pipecat.audio.vad.vad_analyzer import VADParams
from pipecat.frames.frames import EndWorkerFrame, FunctionCallResultProperties, LLMRunFrame
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
from pipecat.services.elevenlabs.tts import ElevenLabsTTSService
from pipecat.services.google.llm import GoogleLLMService
from pipecat.services.llm_service import FunctionCallParams
from pipecat.services.sarvam.stt import SarvamSTTService
from pipecat.services.sarvam.tts import SarvamTTSService
from pipecat.transcriptions.language import Language
from pipecat.transports.base_transport import BaseTransport, TransportParams
from pipecat.workers.runner import WorkerRunner

from brain.config import AppConfig, load_config
from brain.prompt import OPENING_CUE, build_system_prompt
from brain.state import CallState
from brain.tools import TERMINAL_TOOLS, TOOL_SPECS, execute_tool
from voice.latency import attach_latency_logging

load_dotenv(override=True)

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


def build_tts(cfg: AppConfig):
    provider = cfg.tts["provider"]
    if provider == "sarvam":
        c = cfg.tts["sarvam"]
        return SarvamTTSService(
            api_key=os.environ["SARVAM_API_KEY"],
            settings=SarvamTTSService.Settings(
                model=c["model"], voice=c["voice"], language=Language(c["language"])
            ),
        )
    if provider == "deepgram":
        return DeepgramTTSService(
            api_key=os.environ["DEEPGRAM_API_KEY"],
            settings=DeepgramTTSService.Settings(voice=cfg.tts["deepgram"]["voice"]),
        )
    if provider == "elevenlabs":
        c = cfg.tts["elevenlabs"]
        return ElevenLabsTTSService(
            api_key=os.environ["ELEVENLABS_API_KEY"],
            settings=ElevenLabsTTSService.Settings(voice=c["voice_id"], model=c["model"]),
        )
    raise ValueError(f"unknown tts.provider: {provider}")


def build_llm(cfg: AppConfig, customer_name: str):
    b = cfg.brain
    return GoogleLLMService(
        api_key=os.environ["GOOGLE_API_KEY"],
        settings=GoogleLLMService.Settings(
            model=b.agent_model,  # pinned in config.yaml, never Pipecat's default
            system_instruction=build_system_prompt(b, customer_name),
            temperature=b.temperature,
            max_tokens=b.max_output_tokens,
            thinking=GoogleLLMService.ThinkingConfig(thinking_level=b.thinking_level),
        ),
    )


def make_tools_schema() -> ToolsSchema:
    return ToolsSchema(
        standard_tools=[
            FunctionSchema(
                name=s.name, description=s.description, properties=s.properties, required=s.required
            )
            for s in TOOL_SPECS
        ]
    )


def make_tool_handler(state: CallState, name: str):
    async def handler(params: FunctionCallParams):
        result = execute_tool(state, name, params.arguments)
        if name in TERMINAL_TOOLS:
            # The agent already said goodbye in the same reply; do not ask the LLM for more.
            await params.result_callback(
                result, properties=FunctionCallResultProperties(run_llm=False)
            )
            # VERIFY (needs live run): pushed downstream so queued TTS audio flushes first.
            await params.llm.push_frame(EndWorkerFrame(reason=state.end_reason))
        else:
            await params.result_callback(result)

    return handler


async def run_bot(transport: BaseTransport, runner_args: RunnerArguments):
    cfg = load_config(os.getenv("CONFIG_PATH"))
    customer = cfg.brain.default_customer_name
    state = CallState(call_id=uuid.uuid4().hex[:8], customer_name=customer)
    logger.info(f"Starting call {state.call_id} with {customer}")

    stt, tts, llm = build_stt(cfg), build_tts(cfg), build_llm(cfg, customer)
    for spec in TOOL_SPECS:
        llm.register_function(spec.name, make_tool_handler(state, spec.name))

    context = LLMContext(tools=make_tools_schema())
    user_aggregator, assistant_aggregator = LLMContextAggregatorPair(
        context,
        user_params=LLMUserAggregatorParams(
            vad_analyzer=SileroVADAnalyzer(
                params=VADParams(stop_secs=float(cfg.voice["vad_stop_secs"]))
            )
        ),
    )

    pipeline = Pipeline(
        [
            transport.input(),
            stt,
            user_aggregator,
            llm,
            tts,
            transport.output(),
            assistant_aggregator,
        ]
    )

    latency_observer = UserBotLatencyObserver()
    attach_latency_logging(latency_observer, state.call_id, cfg.voice["latency_log"])

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
        logger.info(
            f"Client disconnected: consent={state.consent} callback={state.callback_time} "
            f"dnc={state.dnc} end_reason={state.end_reason}"
        )
        await runner.cancel()

    await runner.run()


async def bot(runner_args: RunnerArguments):
    """Entry point discovered by pipecat.runner."""
    transport = await create_transport(runner_args, transport_params)
    await run_bot(transport, runner_args)


if __name__ == "__main__":
    from pipecat.runner.run import main

    main()

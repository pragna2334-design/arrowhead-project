"""Voice pipeline smoke tests: imports, service construction and the tool-handler hang-up
sequence, all offline (dummy keys, no server, no network)."""

import asyncio
import dataclasses
from types import SimpleNamespace as NS

import pytest
from pipecat.frames.frames import EndWorkerFrame, TTSSpeakFrame
from pipecat.processors.aggregators.llm_context import LLMContext


@pytest.fixture
def bot(monkeypatch):
    for k in ("DEEPGRAM_API_KEY", "SARVAM_API_KEY", "DEEPSEEK_API_KEY", "ELEVENLABS_API_KEY"):
        monkeypatch.setenv(k, "dummy")
    import voice.bot

    return voice.bot


def test_services_construct_for_every_provider(bot, cfg):
    assert type(bot.build_llm(cfg, "Rahul Verma")).__name__ == "DeepSeekLLMService"
    for stt in ("deepgram", "sarvam"):
        c = dataclasses.replace(cfg, stt={**cfg.stt, "provider": stt})
        assert bot.build_stt(c) is not None
    for tts in ("sarvam", "deepgram", "elevenlabs"):
        tts_cfg = {**cfg.tts, "provider": tts, "elevenlabs": {"voice_id": "v", "model": "m"}}
        assert bot.build_tts(dataclasses.replace(cfg, tts=tts_cfg)) is not None
    with pytest.raises(ValueError):
        bot.build_tts(dataclasses.replace(cfg, tts={"provider": "nope"}))


def test_barge_in_and_mute_strategies(bot, cfg):
    p = bot.build_user_params(cfg)
    assert type(p.user_turn_strategies.start[0]).__name__ == "MinWordsUserTurnStartStrategy"
    assert {type(m).__name__ for m in p.user_mute_strategies} == {
        "MuteUntilFirstBotCompleteUserMuteStrategy",
        "FunctionCallUserMuteStrategy",
    }
    p0 = bot.build_user_params(
        dataclasses.replace(cfg, voice={**cfg.voice, "interrupt_min_words": 0})
    )
    assert type(p0.user_turn_strategies.start[0]).__name__ != "MinWordsUserTurnStartStrategy"


class FakeParams:
    def __init__(self, args):
        self.arguments = args
        self.context = LLMContext(
            messages=[
                {"role": "assistant", "content": "Namaste, main Priya, Vriddhi Finance."},
                {"role": "user", "content": "mujhe dobara call mat karna"},
            ]
        )
        self.events: list = []
        self.llm = NS(push_frame=self._push)

    async def _push(self, frame, *a):
        self.events.append(frame)

    async def result_callback(self, result, properties=None):
        self.events.append(("result", result, getattr(properties, "run_llm", None)))


def run_tool(bot, cfg, state, name, args):
    p = FakeParams(args)
    asyncio.run(bot.make_tool_handler(state, name, cfg)(p))
    return p.events


def test_dnc_handler_speaks_closing_line_then_ends(bot, cfg):
    from brain.state import CallState

    state = CallState(call_id="v", customer_name="Rahul Verma")
    ev = run_tool(bot, cfg, state, "mark_do_not_call", {})
    assert ev[0][0] == "result" and ev[0][2] is False  # no further LLM turn
    assert isinstance(ev[1], TTSSpeakFrame) and "hata diya" in ev[1].text
    assert isinstance(ev[2], EndWorkerFrame) and ev[2].reason == "dnc"
    assert state.tool_calls[-1].turn_index == 1  # after the customer's request
    # a second terminal call in the same response must not speak or end again
    ev2 = run_tool(bot, cfg, state, "end_call", {"reason": "declined"})
    assert len(ev2) == 1 and ev2[0][2] is False and state.end_reason == "dnc"


def test_callback_handler_enforces_hours(bot, cfg):
    from brain.state import CallState

    state = CallState(call_id="v", customer_name="Rahul Verma")
    ev = run_tool(bot, cfg, state, "book_callback", {"day": "aaj", "time": "21:00"})
    assert len(ev) == 1 and ev[0][1]["ok"] is False and ev[0][2] is None  # LLM continues
    assert state.callback_time is None and not state.ended

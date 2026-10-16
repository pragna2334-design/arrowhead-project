"""Latency bench, fully offline: stub streaming client, fake clock, fake Sarvam socket."""

import asyncio
import json
from contextlib import asynccontextmanager
from datetime import UTC, datetime
from types import SimpleNamespace as NS

import pytest

from bench import compare
from bench.measure import (
    Pacer,
    SarvamStreamingProbe,
    TTSMeasurement,
    make_splitter,
    measure_llm,
    with_retries,
)
from bench.models import METRICS, BenchReport
from bench.run import default_out_path, greeting_line, load_utterances, main, parse_args, run_bench
from bench.stats import percentile, summarize
from brain.prompt import OPENING_CUE, build_system_prompt


class FakeClock:
    def __init__(self):
        self.t = 0.0

    def __call__(self):
        return self.t


def chunk(content=None, reasoning=None, tool=None):
    tcs = [NS(function=NS(name=tool, arguments=""))] if tool else None
    return NS(choices=[NS(delta=NS(content=content, reasoning_content=reasoning, tool_calls=tcs))])


USAGE = NS(choices=[])  # final usage-only chunk


class FakeStream:
    def __init__(self, clock, script):
        self.clock, self.script = clock, script

    def __aiter__(self):
        return self._gen()

    async def _gen(self):
        for dt, c in self.script:
            self.clock.t += dt
            yield c


class FakeCompletions:
    """create() takes `connect` seconds, then streams the next script (each a [(dt, chunk)])."""

    def __init__(self, clock, scripts, connect=0.1, errors=()):
        self.clock, self.scripts, self.connect = clock, list(scripts), connect
        self.errors = list(errors)
        self.requests: list[dict] = []

    async def create(self, **kw):
        self.requests.append(kw)
        if self.errors:
            raise self.errors.pop(0)
        self.clock.t += self.connect
        return FakeStream(
            self.clock, self.scripts.pop(0) if len(self.scripts) > 1 else self.scripts[0]
        )


def fake_client(clock, scripts, **kw):
    comp = FakeCompletions(clock, scripts, **kw)
    return NS(chat=NS(completions=comp)), comp


# A typical reply: role chunk at 180 ms, then the first content token 600 ms later.
REPLY = [
    (0.08, chunk()),  # role-only delta
    (0.6, chunk("Haan")),
    (0.02, chunk(" ji")),
    (0.02, chunk(",")),
    (0.02, chunk(" main")),
    (0.02, chunk(" samajh")),
    (0.02, chunk(" sakti")),
    (0.02, chunk(" hoon")),
    (0.02, chunk(".")),
    (0.02, chunk(" Rate")),
    (0.05, USAGE),
]


def run_measure(script, mode="clause", connect=0.1):
    clock = FakeClock()
    client, comp = fake_client(clock, [script], connect=connect)
    m = asyncio.run(
        measure_llm(
            client,
            model="deepseek-flash",
            messages=[{"role": "user", "content": "x"}],
            tools=[{"type": "function"}],
            temperature=0.4,
            max_tokens=300,
            thinking="disabled",
            splitter=make_splitter(mode, "hi-IN"),
            clock=clock,
        )
    )
    return m, comp.requests[0]


def test_llm_metrics_with_fake_clock():
    m, req = run_measure(REPLY)
    assert m.ttfb_ms == 180.0  # connect 100 + first chunk 80
    assert m.first_text_ms == 780.0
    assert m.first_clause == "Haan ji," and m.first_clause_ms == 840.0  # released at " main"
    assert m.total_ms == 990.0
    assert m.reasoning_chunks == 0 and not m.tool_call
    assert m.text == "Haan ji, main samajh sakti hoon. Rate"
    # same request shape as the agent / Pipecat service
    assert req["stream"] is True and req["extra_body"] == {"thinking": {"type": "disabled"}}
    assert req["model"] == "deepseek-flash" and req["max_tokens"] == 300 and req["tools"]


def test_sentence_and_token_modes_cut_differently():
    s, _ = run_measure(REPLY, mode="sentence")
    assert (
        s.first_clause == "Haan ji, main samajh sakti hoon." and s.first_clause_ms == 940.0
    )  # at " Rate"
    t, _ = run_measure(REPLY, mode="token")
    assert t.first_clause == "Haan" and t.first_clause_ms == t.first_text_ms == 780.0


def test_reasoning_chunks_and_tool_calls_detected():
    script = [
        (0.05, chunk(reasoning="Customer wants")),
        (0.05, chunk(reasoning=" a callback")),
        (0.3, chunk(tool="book_callback")),
        (0.01, chunk()),
        (0.01, USAGE),
    ]
    m, _ = run_measure(script)
    assert m.reasoning_chunks == 2 and m.tool_call and m.tool_names == ["book_callback"]
    assert m.first_text_ms is None and m.first_clause is None and m.first_clause_ms is None


def test_short_reply_released_at_end_of_stream():
    m, _ = run_measure([(0.2, chunk("Ji")), (0.1, chunk(" haan")), (0.05, USAGE)])
    assert m.first_clause == "Ji haan" and m.first_clause_ms == m.total_ms == 450.0


def test_percentile_linear_interpolation():
    assert percentile([], 50) is None
    assert percentile([5], 95) == 5
    assert percentile([1, 2, 3, 4], 50) == 2.5
    assert percentile([10, 20, 30, 40, 50], 95) == pytest.approx(48.0)
    s = summarize([100, 200, 300])
    assert (s.n, s.p50, s.p95, s.mean, s.min, s.max) == (3, 200, 290, 200, 100, 300)
    with pytest.raises(ValueError):
        percentile([1], 101)


def test_with_retries_backs_off_on_429_only():
    sleeps: list[float] = []

    async def sleep(d):
        sleeps.append(d)

    class RateLimited(Exception):
        status_code = 429

    calls = {"n": 0}

    async def flaky():
        calls["n"] += 1
        if calls["n"] < 3:
            raise RateLimited()
        return "ok"

    assert asyncio.run(with_retries(flaky, sleep=sleep, delays=(1, 2, 4))) == "ok"
    assert sleeps == [1, 2]

    async def bad():
        raise ValueError("not transient")

    with pytest.raises(ValueError):
        asyncio.run(with_retries(bad, sleep=sleep))

    async def always():
        raise RateLimited()

    with pytest.raises(RateLimited):
        asyncio.run(with_retries(always, sleep=sleep, delays=(1,)))


def test_pacer_spaces_requests():
    clock, sleeps = FakeClock(), []

    async def sleep(d):
        sleeps.append(round(d, 3))
        clock.t += d

    async def go():
        p = Pacer(1.0, clock=clock, sleep=sleep)
        await p.wait()  # first request: no wait
        clock.t += 0.25
        await p.wait()
        clock.t += 2.0
        await p.wait()  # already more than 1 s later

    asyncio.run(go())
    assert sleeps == [0.75]


# --- Sarvam probe ------------------------------------------------------------------------------


class FakeSocket:
    def __init__(self, clock, messages, delay=0.17):
        self.clock, self.messages, self.delay = clock, messages, delay
        self.sent: list = []

    async def _send_model(self, m):
        self.sent.append(("config", m.dict()))

    async def convert(self, text):
        self.sent.append(("text", text))

    async def flush(self):
        self.sent.append(("flush", None))

    def __aiter__(self):
        return self._gen()

    async def _gen(self):
        for msg in self.messages:
            self.clock.t += self.delay
            yield msg


def fake_sarvam(clock, sock, connect=0.3):
    calls = {}

    @asynccontextmanager
    async def connect_(**kw):
        calls.update(kw)
        clock.t += connect
        yield sock

    return NS(text_to_speech_streaming=NS(connect=connect_)), calls


def test_sarvam_probe_times_first_audio_and_mirrors_pipecat_config():
    clock = FakeClock()
    sock = FakeSocket(clock, [NS(type="event"), NS(type="audio", data=NS(audio="AAA"))])
    client, calls = fake_sarvam(clock, sock)
    probe = SarvamStreamingProbe(
        api_key="k", model="bulbul:v3", voice="priya", language="hi-IN", client=client, clock=clock
    )
    t = asyncio.run(probe.first_audio("Haan ji,"))
    assert t.connect_ms == 300.0 and t.ttfb_ms == 340.0  # two messages, audio on the second
    assert calls == {"model": "bulbul:v3", "send_completion_event": True}
    (kind, cfg_msg), *rest = sock.sent
    assert kind == "config" and cfg_msg["type"] == "config"
    data = cfg_msg["data"]
    assert data["speaker"] == "priya" and data["target_language_code"] == "hi-IN"
    assert data["min_buffer_size"] == 50 and "pitch" not in data and "loudness" not in data
    assert rest == [("text", "Haan ji,"), ("flush", None)]


def test_sarvam_probe_raises_on_error_message():
    clock = FakeClock()
    sock = FakeSocket(clock, [NS(type="error", data=NS(message="bad speaker"))])
    client, _ = fake_sarvam(clock, sock)
    probe = SarvamStreamingProbe(
        api_key="k", model="bulbul:v3", voice="x", language="hi-IN", client=client, clock=clock
    )
    with pytest.raises(RuntimeError, match="bad speaker"):
        asyncio.run(probe.first_audio("Haan ji,"))


# --- full run --------------------------------------------------------------------------------


class FakeProbe:
    def __init__(self):
        self.texts: list[str] = []

    async def first_audio(self, text):
        self.texts.append(text)
        return TTSMeasurement(ttfb_ms=175.0, connect_ms=300.0)


async def _no_sleep(_d):
    return None


NOW = datetime(2026, 10, 16, 12, 0, 0, tzinfo=UTC)


def bench(cfg, argv, *, client, probe=None, clock=None):
    return asyncio.run(
        run_bench(
            parse_args(argv),
            cfg=cfg,
            client=client,
            tts_probe=probe,
            clock=clock or FakeClock(),
            sleep=_no_sleep,
            now=NOW,
        )
    )


def test_run_writes_valid_json_with_metadata_and_stats(cfg, tmp_path, monkeypatch):
    monkeypatch.setenv("SARVAM_API_KEY", "dummy")
    clock = FakeClock()
    client, comp = fake_client(clock, [REPLY])
    probe = FakeProbe()
    out = tmp_path / "b.json"
    _, path = bench(
        cfg,
        ["--utterances", "2", "--repeats", "2", "--label", "after", "--out", str(out)],
        client=client,
        probe=probe,
        clock=clock,
    )
    assert path == out
    data = json.loads(out.read_text())
    parsed = BenchReport.model_validate(data)
    meta = parsed.meta
    assert (meta.label, meta.model, meta.thinking) == ("after", cfg.brain.agent_model, "disabled")
    assert meta.prompt_version == cfg.brain.prompt_version and meta.scenario == cfg.scenario.id
    assert meta.aggregation == "clause" and meta.utterances == 2 and meta.repeats == 2
    assert meta.tts.model == "bulbul:v3" and meta.tts_skipped_reason is None
    assert meta.created_at == "2026-10-16T12:00:00+00:00"
    assert len(parsed.samples) == 4 and set(parsed.stats) == set(METRICS)
    s = parsed.samples[0]
    assert s.utterance_id == "rate_high" and s.first_clause == "Haan ji,"
    assert s.tts_ttfb_ms == 175.0 and s.est_first_audio_ms == 840.0 + 175.0
    assert parsed.stats["est_first_audio_ms"].p50 == 1015.0 and parsed.stats["llm_ttfb_ms"].n == 4
    assert probe.texts == ["Haan ji,"] * 4

    # Request = the agent's: real system prompt, opening cue, greeting, mood-tagged customer line
    req = comp.requests[0]
    msgs = req["messages"]
    assert [m["role"] for m in msgs] == ["system", "user", "assistant", "user"]
    assert msgs[0]["content"] == build_system_prompt(cfg.brain, cfg.brain.default_customer_name)
    assert msgs[1]["content"] == OPENING_CUE
    assert msgs[2]["content"] == greeting_line(cfg, cfg.brain.default_customer_name)
    assert (
        msgs[3]["content"].startswith("Rate bahut zyada hai")
        and "customer_mood" in msgs[3]["content"]
    )
    assert req["temperature"] == cfg.brain.temperature
    assert req["max_tokens"] == cfg.brain.max_output_tokens
    assert {t["function"]["name"] for t in req["tools"]} >= {"end_call", "book_callback"}


def test_tts_skipped_without_sarvam_key(cfg, tmp_path, monkeypatch):
    monkeypatch.delenv("SARVAM_API_KEY", raising=False)
    from loguru import logger

    warnings: list[str] = []
    sink = logger.add(lambda m: warnings.append(m.record["message"]), level="WARNING")
    try:
        clock = FakeClock()
        client, _ = fake_client(clock, [REPLY])
        report, _ = bench(
            cfg,
            ["--utterances", "1", "--repeats", "1", "--out", str(tmp_path / "x.json")],
            client=client,
            clock=clock,
        )
    finally:
        logger.remove(sink)
    assert report.meta.tts is None and report.meta.tts_skipped_reason == "SARVAM_API_KEY not set"
    assert report.samples[0].tts_ttfb_ms is None and report.samples[0].est_first_audio_ms is None
    assert report.stats["tts_ttfb_ms"].n == 0 and report.samples[0].llm_first_clause_ms == 840.0
    assert any("SARVAM_API_KEY" in w for w in warnings)


def test_no_tts_flag_skips_probe(cfg, tmp_path):
    client, _ = fake_client(FakeClock(), [REPLY])
    probe = FakeProbe()
    report, _ = bench(
        cfg,
        ["--utterances", "1", "--repeats", "1", "--no-tts", "--out", str(tmp_path / "x.json")],
        client=client,
        probe=probe,
    )
    assert report.meta.tts_skipped_reason == "--no-tts" and probe.texts == []


def test_overrides_and_failed_samples(cfg, tmp_path):
    class Boom(Exception):
        status_code = 400

    client, comp = fake_client(FakeClock(), [REPLY], errors=[Boom("bad request")])
    report, _ = bench(
        cfg,
        [
            "--utterances", "1", "--repeats", "2", "--aggregation", "sentence",
            "--thinking", "enabled", "--version", "v3", "--out", str(tmp_path / "x.json"),
        ],
        client=client,
    )  # fmt: skip
    assert report.meta.aggregation == "sentence" and report.meta.thinking == "enabled"
    assert comp.requests[-1]["extra_body"] == {"thinking": {"type": "enabled"}}
    assert report.samples[0].error == "Boom: bad request" and report.samples[1].error is None
    assert report.stats["llm_ttfb_ms"].n == 1  # failed samples are excluded from stats
    assert report.samples[1].first_clause == "Haan ji, main samajh sakti hoon."


def test_missing_deepseek_key_exits_2(monkeypatch, capsys, tmp_path):
    monkeypatch.delenv("DEEPSEEK_API_KEY", raising=False)
    assert main(["--out", str(tmp_path / "x.json")]) == 2
    assert "DEEPSEEK_API_KEY" in capsys.readouterr().err
    assert not (tmp_path / "x.json").exists()


def test_bad_prompt_version_exits_2(capsys, tmp_path):
    assert main(["--version", "v99", "--out", str(tmp_path / "x.json")]) == 2
    assert "v99" in capsys.readouterr().err


def test_main_prints_summary_and_reasoning_warning(cfg, tmp_path, capsys, monkeypatch):
    monkeypatch.delenv("SARVAM_API_KEY", raising=False)
    clock = FakeClock()
    client, _ = fake_client(clock, [[(0.1, chunk(reasoning="hmm")), *REPLY]])
    out = tmp_path / "r.json"
    code = main(
        ["--utterances", "1", "--repeats", "1", "--out", str(out)],
        cfg=cfg,
        client=client,
        clock=clock,
        sleep=_no_sleep,
    )
    text = capsys.readouterr().out
    assert code == 0 and "llm_first_clause_ms" in text and "thinking is NOT really off" in text


def test_utterances_file_and_default_path():
    utts = load_utterances()
    assert len(utts) >= 15 and len({u["id"] for u in utts}) == len(utts)
    assert {u["kind"] for u in utts} == {"objection", "question", "callback"}
    p = default_out_path("logs/bench", "before / v3", NOW)
    assert str(p) == "logs/bench/before-v3-20261016T120000Z.json"


# --- compare ---------------------------------------------------------------------------------

FIXTURES = __import__("pathlib").Path(__file__).parent / "fixtures" / "bench"


def test_compare_prints_p50_p95_deltas(capsys):
    assert compare.main([str(FIXTURES / "before.json"), str(FIXTURES / "after.json")]) == 0
    out = capsys.readouterr().out
    row = next(line for line in out.splitlines() if line.startswith("llm_first_clause_ms"))
    assert row.split()[:4] == ["llm_first_clause_ms", "1000", "800", "-200"]
    assert "(-20%)" in row
    agg = next(line for line in out.splitlines() if line.startswith("aggregation"))
    assert "sentence" in agg and "clause" in agg and agg.rstrip().endswith("*")
    tts = next(line for line in out.splitlines() if line.startswith("tts_ttfb_ms"))
    assert tts.split()[1:3] == ["175", "-"]  # metric missing in the after run


def test_compare_missing_file_exits_2(capsys, tmp_path):
    assert compare.main([str(tmp_path / "nope.json"), str(FIXTURES / "after.json")]) == 2
    assert "error" in capsys.readouterr().err

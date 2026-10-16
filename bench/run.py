"""Latency bench: how soon could the agent start speaking? (`make bench`)

    python -m bench.run [--utterances N] [--repeats R] [--label before] [--out PATH]
                        [--no-tts] [--scenario ID] [--version vN]
                        [--aggregation sentence|clause|token] [--thinking enabled|disabled]

For each fixed customer line (bench/utterances.yaml) x repeat it streams one DeepSeek chat
completion exactly as the agent would (real system prompt, opening cue, greeting, mood-tagged
customer line, same model / temperature / max_tokens / thinking / tools) and records:
time to first chunk, first content token, first chunk the TTS text aggregator would release,
stream end, reasoning chunks and tool calls. The first released chunk is then sent to Sarvam
TTS (websocket) to time the first audio byte; est_first_audio_ms = first clause + TTS TTFB.

Exit codes: 0 ok, 1 every sample failed, 2 setup error (e.g. DEEPSEEK_API_KEY missing).
"""

from __future__ import annotations

import argparse
import asyncio
import os
import re
import sys
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import yaml
from loguru import logger

from bench.measure import (
    Clock,
    Pacer,
    SarvamStreamingProbe,
    Sleep,
    TTSProbe,
    make_splitter,
    measure_llm,
    with_retries,
)
from bench.models import BenchMeta, BenchReport, BenchSample, TTSMeta
from bench.stats import stats_for
from brain.config import AppConfig, load_config
from brain.prompt import OPENING_CUE, build_system_prompt
from brain.sentiment import tag_customer_text
from voice.chunking import DEFAULT_AGGREGATION, parse_aggregation_mode
from voice.llm_settings import THINKING_VALUES, log_thinking, validate_thinking

UTTERANCES_PATH = Path(__file__).resolve().parent / "utterances.yaml"
SARVAM_SAMPLE_RATE = 24000  # bulbul:v3 default in Pipecat's SarvamTTSService


class SetupError(Exception):
    """Bad arguments / missing keys: exit 2."""


def load_utterances(path: str | Path = UTTERANCES_PATH) -> list[dict[str, str]]:
    data = yaml.safe_load(Path(path).read_text())
    items = data.get("utterances") if isinstance(data, dict) else None
    if not items:
        raise SetupError(f"no utterances in {path}")
    for u in items:
        if not u.get("id") or not u.get("text"):
            raise SetupError(f"utterance without id/text in {path}: {u}")
    return items


def greeting_line(cfg: AppConfig, customer_name: str) -> str:
    """The scenario's opening line, rendered like prompts render opening_example."""
    verb = "bol rahi hoon" if cfg.brain.agent_gender == "female" else "bol raha hoon"
    return cfg.scenario.opening.format(
        agent_name=cfg.brain.agent_name,
        verb=verb,
        company=cfg.scenario.company,
        customer_name=customer_name,
    )


def build_messages(system: str, greeting: str, utterance: str) -> tuple[list[dict], str]:
    tagged, mood = tag_customer_text(utterance)
    return (
        [
            {"role": "system", "content": system},
            {"role": "user", "content": OPENING_CUE},
            {"role": "assistant", "content": greeting},
            {"role": "user", "content": tagged},
        ],
        mood,
    )


def default_out_path(out_dir: str | Path, label: str, created: datetime) -> Path:
    safe = re.sub(r"[^A-Za-z0-9_.-]+", "-", label).strip("-") or "run"
    return Path(out_dir) / f"{safe}-{created.strftime('%Y%m%dT%H%M%SZ')}.json"


def parse_args(argv: list[str] | None) -> argparse.Namespace:
    p = argparse.ArgumentParser(prog="python -m bench.run", description=__doc__.split("\n")[0])
    p.add_argument("--utterances", type=int, help="first N utterances (config bench.utterances)")
    p.add_argument("--repeats", type=int, help="runs per utterance (config bench.repeats)")
    p.add_argument("--label", default="run", help="name for this run, e.g. before / after")
    p.add_argument("--out", help="output JSON (default <bench.out_dir>/<label>-<ts>.json)")
    p.add_argument("--no-tts", action="store_true", help="skip the Sarvam TTS measurement")
    p.add_argument("--scenario", help="scenario id (default: config / SCENARIO)")
    p.add_argument("--version", help="prompt version (default: config / PROMPT_VERSION)")
    p.add_argument(
        "--aggregation",
        choices=("sentence", "clause", "token"),
        help="how to cut the first chunk (default tts.sarvam.text_aggregation)",
    )
    p.add_argument("--thinking", choices=THINKING_VALUES, help="override brain.thinking")
    p.add_argument("--interval", type=float, default=1.0, help="seconds between LLM requests")
    p.add_argument("--utterances-file", default=str(UTTERANCES_PATH), help=argparse.SUPPRESS)
    return p.parse_args(argv)


def _make_llm_client() -> Any:
    from brain.agent import make_client  # same client as text mode (max_retries=0)

    return make_client()


def _make_tts_probe(cfg: AppConfig, clock: Clock) -> TTSProbe:
    c = cfg.tts.get("sarvam", {})
    return SarvamStreamingProbe(
        api_key=os.environ["SARVAM_API_KEY"],
        model=c.get("model", "bulbul:v3"),
        voice=c.get("voice", "priya"),
        language=c.get("language", "hi-IN"),
        min_buffer_size=int(c.get("min_buffer_size") or 50),
        sample_rate=SARVAM_SAMPLE_RATE,
        clock=clock,
    )


async def run_bench(
    args: argparse.Namespace,
    *,
    cfg: AppConfig | None = None,
    client: Any = None,
    tts_probe: TTSProbe | None = None,
    clock: Clock = time.perf_counter,
    sleep: Sleep = asyncio.sleep,
    now: datetime | None = None,
) -> tuple[BenchReport, Path]:
    """Run the bench and write the JSON. Raises SetupError for exit-2 problems."""
    if cfg is None:
        try:
            cfg = load_config(prompt_version=args.version, scenario=args.scenario)
        except (FileNotFoundError, ValueError) as e:
            raise SetupError(str(e)) from e
    b = cfg.brain
    try:
        thinking = validate_thinking(args.thinking or b.thinking)
        sarvam = cfg.tts.get("sarvam", {})
        aggregation = parse_aggregation_mode(
            args.aggregation or sarvam.get("text_aggregation", DEFAULT_AGGREGATION)
        )
        customer = b.default_customer_name
        system = build_system_prompt(b, customer)
    except (FileNotFoundError, ValueError) as e:
        raise SetupError(str(e)) from e

    if client is None:
        if not os.environ.get("DEEPSEEK_API_KEY"):
            raise SetupError("DEEPSEEK_API_KEY is not set (put it in .env); the bench needs it")
        client = _make_llm_client()

    skipped: str | None = None
    tts_meta: TTSMeta | None = None
    if args.no_tts:
        skipped, tts_probe = "--no-tts", None
    elif tts_probe is None:
        if os.environ.get("SARVAM_API_KEY"):
            tts_probe = _make_tts_probe(cfg, clock)
        else:
            skipped = "SARVAM_API_KEY not set"
            logger.warning("SARVAM_API_KEY not set: skipping TTS (tts_ttfb_ms will be empty)")
    if tts_probe is not None:
        tts_meta = TTSMeta(
            model=sarvam.get("model", "bulbul:v3"),
            voice=sarvam.get("voice", "priya"),
            language=sarvam.get("language", "hi-IN"),
            min_buffer_size=int(sarvam.get("min_buffer_size") or 50),
            sample_rate=SARVAM_SAMPLE_RATE,
        )

    bench_cfg = cfg.bench or {}
    utterances = load_utterances(args.utterances_file)
    n = args.utterances or int(bench_cfg.get("utterances", len(utterances)))
    repeats = args.repeats or int(bench_cfg.get("repeats", 1))
    if n < 1 or repeats < 1:
        raise SetupError("--utterances and --repeats must be >= 1")
    utterances = utterances[:n]

    log_thinking(b.agent_model, thinking)
    logger.info(
        f"Bench '{args.label}': {len(utterances)} utterances x {repeats}, prompt "
        f"{b.prompt_version}, scenario {cfg.scenario.id}, aggregation={aggregation}, "
        f"tts={'skipped (' + skipped + ')' if skipped else 'sarvam'}"
    )

    from brain.agent import openai_tools

    tools = openai_tools(cfg.scenario)
    greeting = greeting_line(cfg, customer)
    language = sarvam.get("language")
    pacer = Pacer(args.interval, sleep=sleep)

    def on_retry(e: BaseException, delay: float) -> None:
        logger.warning(f"transient error ({e}); retrying in {delay:.0f}s")

    samples: list[BenchSample] = []
    for u in utterances:
        messages, mood = build_messages(system, greeting, u["text"])
        for r in range(repeats):
            sample = BenchSample(utterance_id=u["id"], repeat=r, mood=mood)

            async def one_llm(messages=messages):
                await pacer.wait()
                return await measure_llm(
                    client,
                    model=b.agent_model,
                    messages=messages,
                    tools=tools,
                    temperature=b.temperature,
                    max_tokens=b.max_output_tokens,
                    thinking=thinking,
                    splitter=make_splitter(aggregation, language),
                    clock=clock,
                )

            try:
                m = await with_retries(one_llm, sleep=sleep, on_retry=on_retry)
                sample = sample.model_copy(
                    update={
                        "llm_ttfb_ms": m.ttfb_ms,
                        "llm_first_text_ms": m.first_text_ms,
                        "llm_first_clause_ms": m.first_clause_ms,
                        "llm_total_ms": m.total_ms,
                        "reasoning_chunks": m.reasoning_chunks,
                        "tool_call": m.tool_call,
                        "tool_names": m.tool_names,
                        "first_clause": m.first_clause,
                        "first_clause_chars": len(m.first_clause) if m.first_clause else None,
                        "reply_text": m.text,
                    }
                )
                if tts_probe is not None and m.first_clause:
                    t = await with_retries(
                        lambda probe=tts_probe, text=m.first_clause: probe.first_audio(text),
                        sleep=sleep,
                        on_retry=on_retry,
                    )
                    sample = sample.model_copy(
                        update={
                            "tts_ttfb_ms": t.ttfb_ms,
                            "tts_connect_ms": t.connect_ms,
                            "est_first_audio_ms": round(m.first_clause_ms + t.ttfb_ms, 1)
                            if m.first_clause_ms is not None
                            else None,
                        }
                    )
            except Exception as e:  # noqa: BLE001 - one bad sample must not stop the run
                sample = sample.model_copy(update={"error": f"{type(e).__name__}: {e}"})
                logger.error(f"{u['id']}#{r}: {sample.error}")
            samples.append(sample)
            logger.info(
                f"{u['id']}#{r}: ttfb={sample.llm_ttfb_ms} first_text={sample.llm_first_text_ms} "
                f"first_clause={sample.llm_first_clause_ms} tts={sample.tts_ttfb_ms} "
                f"reasoning={sample.reasoning_chunks} tools={sample.tool_names}"
            )

    created = now or datetime.now(UTC)
    report = BenchReport(
        meta=BenchMeta(
            label=args.label,
            created_at=created.isoformat(timespec="seconds"),
            model=b.agent_model,
            thinking=thinking,
            temperature=b.temperature,
            max_tokens=b.max_output_tokens,
            prompt_version=b.prompt_version,
            scenario=cfg.scenario.id,
            aggregation=aggregation,
            utterances=len(utterances),
            repeats=repeats,
            tts=tts_meta,
            tts_skipped_reason=skipped,
        ),
        samples=samples,
        stats=stats_for(samples),
    )
    out = (
        Path(args.out)
        if args.out
        else default_out_path(bench_cfg.get("out_dir", "logs/bench"), args.label, created)
    )
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(report.model_dump_json(indent=2))
    return report, out


def print_summary(report: BenchReport, out: Path) -> None:
    print(f"\nBench '{report.meta.label}' -> {out}")
    print(f"{'metric':<22}{'n':>4}{'p50':>10}{'p95':>10}")
    for name, s in report.stats.items():
        p50 = "-" if s.p50 is None else f"{s.p50:.0f}"
        p95 = "-" if s.p95 is None else f"{s.p95:.0f}"
        print(f"{name:<22}{s.n:>4}{p50:>10}{p95:>10}")
    if any(s.reasoning_chunks for s in report.samples):
        print("WARNING: reasoning chunks streamed: thinking is NOT really off for this model.")


def main(argv: list[str] | None = None, **deps: Any) -> int:
    args = parse_args(argv)
    try:
        report, out = asyncio.run(run_bench(args, **deps))
    except SetupError as e:
        print(f"error: {e}", file=sys.stderr)
        return 2
    print_summary(report, out)
    return 1 if report.samples and all(s.error for s in report.samples) else 0


if __name__ == "__main__":
    from dotenv import load_dotenv

    load_dotenv(override=True)
    sys.exit(main())

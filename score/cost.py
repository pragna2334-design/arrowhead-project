"""Per-call cost estimate from token usage, audio seconds and the config price table.

    cost_record(t, judge_usage, cfg) -> CostRecord

Prices come from config.yaml `prices` (ALL placeholders to VERIFY against your invoices):
- LLM: USD per 1M tokens by model id, split into uncached input, cached input and output.
  Agent tokens are priced at the transcript's agent model, simulated-customer tokens at
  `sim.customer_model`, judge tokens at `brain.judge_model`.
- STT: USD per audio minute of customer speech, by `stt.provider`.
- TTS: USD per 1k characters the agent spoke, by `tts.provider`.

Voice transcripts carry measured audio seconds (`Transcript.audio_seconds`). Text-mode runs have
no audio, so seconds are estimated from text (`prices.estimate`) and `audio_estimated` is set.

production_usd = llm_agent + stt + tts           (what one real call would cost)
eval_usd       = production + llm_customer + llm_judge   (what this simulated eval cost)

An unknown model / provider prices at 0 with a warning; it never raises.
"""

from __future__ import annotations

import sys
from collections.abc import Iterable, Mapping
from typing import Any

from schemas import CallScore, CostRecord, CostSummary, Transcript, Usage

BREAKDOWN_KEYS = ("llm_agent", "llm_customer", "llm_judge", "stt", "tts")
DEFAULT_TTS_CHARS_PER_SECOND = 14.0
DEFAULT_STT_WORDS_PER_SECOND = 2.5

_warned: set[str] = set()


def _warn(msg: str) -> None:
    if msg not in _warned:  # once per process per distinct problem
        _warned.add(msg)
        print(f"warning: cost: {msg}", file=sys.stderr)


def _num(d: Mapping[str, Any] | None, key: str) -> float | None:
    if not isinstance(d, Mapping) or d.get(key) is None:
        return None
    try:
        return float(d[key])
    except (TypeError, ValueError):
        return None


def llm_usd(usage: Usage, price: Mapping[str, Any] | None) -> float:
    """USD for one role's tokens. Cached prompt tokens are billed at `cached_input` (falls back
    to `input` when the table has no cached price); the rest of the prompt at `input`."""
    if not price or usage.requests == 0 and usage.prompt_tokens == 0:
        return 0.0
    p_in = _num(price, "input") or 0.0
    p_cached = _num(price, "cached_input")
    p_cached = p_in if p_cached is None else p_cached
    p_out = _num(price, "output") or 0.0
    cached = min(usage.cached_prompt_tokens, usage.prompt_tokens)
    uncached = usage.prompt_tokens - cached
    return (uncached * p_in + cached * p_cached + usage.completion_tokens * p_out) / 1_000_000


def _llm_price(prices: Mapping[str, Any], model: str | None, role: str) -> Mapping[str, Any]:
    table = prices.get("llm") or {}
    price = table.get(model) if model else None
    if not isinstance(price, Mapping):
        _warn(f"no LLM price for {role} model {model!r} in prices.llm; priced at 0")
        return {}
    return price


def _provider_price(prices: Mapping[str, Any], kind: str, provider: str | None, key: str) -> float:
    entry = (prices.get(kind) or {}).get(provider) if provider else None
    value = _num(entry, key)
    if value is None:
        _warn(f"no {kind} price {key!r} for provider {provider!r} in prices.{kind}; priced at 0")
        return 0.0
    return value


def _cfg_parts(
    cfg: Any,
) -> tuple[Mapping[str, Any], str | None, str | None, str | None, str | None]:
    """(prices, customer_model, judge_model, stt_provider, tts_provider) from an AppConfig."""
    prices = getattr(cfg, "prices", None) or {}
    sim = getattr(cfg, "sim", None) or {}
    brain = getattr(cfg, "brain", None)
    stt = getattr(cfg, "stt", None) or {}
    tts = getattr(cfg, "tts", None) or {}
    return (
        prices,
        sim.get("customer_model"),
        getattr(brain, "judge_model", None),
        stt.get("provider"),
        tts.get("provider"),
    )


def cost_record(t: Transcript, judge_usage: Usage | None, cfg: Any) -> CostRecord:
    """Price one call. Old transcripts without `usage` cost 0 for LLM/audio (see has_usage)."""
    prices, customer_model, judge_model, stt_provider, tts_provider = _cfg_parts(cfg)
    usage = t.usage
    agent_u = usage.agent if usage else Usage()
    customer_u = usage.customer if usage else Usage()
    judge_u = judge_usage or Usage()
    tts_chars = usage.agent_chars if usage else 0
    customer_words = usage.customer_words if usage else 0

    audio = t.audio_seconds or {}
    if t.mode == "voice" and "stt" in audio and "tts" in audio:
        stt_s, tts_s, estimated = float(audio["stt"]), float(audio["tts"]), False
    else:
        est = prices.get("estimate") or {}
        cps = _num(est, "tts_chars_per_second") or DEFAULT_TTS_CHARS_PER_SECOND
        wps = _num(est, "stt_words_per_second") or DEFAULT_STT_WORDS_PER_SECOND
        tts_s, stt_s, estimated = tts_chars / cps, customer_words / wps, True

    breakdown = {
        "llm_agent": llm_usd(agent_u, _llm_price(prices, t.agent_model, "agent"))
        if agent_u.requests or agent_u.prompt_tokens
        else 0.0,
        "llm_customer": llm_usd(customer_u, _llm_price(prices, customer_model, "customer"))
        if customer_u.requests or customer_u.prompt_tokens
        else 0.0,
        "llm_judge": llm_usd(judge_u, _llm_price(prices, judge_model, "judge"))
        if judge_u.requests or judge_u.prompt_tokens
        else 0.0,
        "stt": stt_s / 60 * _provider_price(prices, "stt", stt_provider, "per_minute")
        if stt_s
        else 0.0,
        "tts": tts_chars / 1000 * _provider_price(prices, "tts", tts_provider, "per_1k_chars")
        if tts_chars
        else 0.0,
    }
    breakdown = {k: round(v, 10) for k, v in breakdown.items()}
    production = breakdown["llm_agent"] + breakdown["stt"] + breakdown["tts"]
    eval_usd = production + breakdown["llm_customer"] + breakdown["llm_judge"]
    return CostRecord(
        call_id=t.call_id,
        scenario=t.scenario,
        prompt_version=t.prompt_version,
        mode=t.mode,
        agent_usage=agent_u,
        customer_usage=customer_u,
        judge_usage=judge_u,
        stt_seconds=round(stt_s, 3),
        tts_seconds=round(tts_s, 3),
        tts_chars=tts_chars,
        audio_estimated=estimated,
        breakdown_usd=breakdown,
        production_usd=round(production, 10),
        eval_usd=round(eval_usd, 10),
        price_table=str(prices.get("table") or "unknown"),
    )


def has_usage(c: CostRecord) -> bool:
    """False for calls priced from an old transcript without recorded usage (cost is 0)."""
    return c.agent_usage.requests > 0 or c.agent_usage.prompt_tokens > 0 or c.tts_chars > 0


def summarize_costs(scores: Iterable[CallScore]) -> CostSummary | None:
    """Roll CallScore.cost up into a CostSummary. None when no call has a cost record."""
    calls = list(scores)
    costed = [s for s in calls if s.cost is not None]
    if not costed:
        return None
    n = len(costed)
    prod_total = sum(s.cost.production_usd for s in costed)
    eval_total = sum(s.cost.eval_usd for s in costed)
    booked = sum(1 for s in costed if s.outcome == "callback_booked")
    tables = sorted({s.cost.price_table for s in costed})
    keys = list(BREAKDOWN_KEYS) + sorted(
        {k for s in costed for k in s.cost.breakdown_usd} - set(BREAKDOWN_KEYS)
    )
    return CostSummary(
        price_table=tables[0] if len(tables) == 1 else "mixed: " + ", ".join(tables),
        calls=n,
        production_usd_mean=round(prod_total / n, 6),
        production_usd_total=round(prod_total, 6),
        eval_usd_total=round(eval_total, 6),
        booked_callbacks=booked,
        production_usd_per_booked_callback=round(prod_total / booked, 6) if booked else None,
        breakdown_usd_mean={
            k: round(sum(s.cost.breakdown_usd.get(k, 0.0) for s in costed) / n, 6) for k in keys
        },
    )

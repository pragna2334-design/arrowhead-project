"""Offline tests for score/cost.py: per-call pricing and the cost summary."""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace as NS

import pytest

from schemas import CallUsage, Transcript, Usage
from score.cost import cost_record, has_usage, llm_usd, summarize_costs

FX = Path(__file__).parent / "fixtures" / "score"

PRICES = {
    "table": "test-table",
    "llm": {
        "agent-m": {"input": 0.27, "cached_input": 0.07, "output": 1.10},
        "cust-m": {"input": 0.27, "output": 1.10},  # no cached price: falls back to input
        "judge-m": {"input": 0.55, "cached_input": 0.14, "output": 2.19},
    },
    "stt": {"deepgram": {"per_minute": 0.0092}},
    "tts": {"sarvam": {"per_1k_chars": 0.018}},
    "estimate": {"tts_chars_per_second": 14.0, "stt_words_per_second": 2.5},
}


def cfg(**over):
    base = {
        "prices": PRICES,
        "sim": {"customer_model": "cust-m"},
        "brain": NS(judge_model="judge-m", agent_model="agent-m"),
        "stt": {"provider": "deepgram"},
        "tts": {"provider": "sarvam"},
    }
    base.update(over)
    return NS(**base)


def transcript(**update) -> Transcript:
    t = Transcript.model_validate_json((FX / "compliant.json").read_text())
    usage = CallUsage(
        agent=Usage(
            requests=5, prompt_tokens=10_000, completion_tokens=500, cached_prompt_tokens=4000
        ),
        customer=Usage(
            requests=4, prompt_tokens=4_000, completion_tokens=200, cached_prompt_tokens=1000
        ),
        agent_chars=1400,
        customer_words=50,
    )
    return t.model_copy(update={"agent_model": "agent-m", "usage": usage, **update})


JUDGE = Usage(requests=1, prompt_tokens=2_000, completion_tokens=300)


def test_llm_usd_cached_split():
    u = Usage(requests=1, prompt_tokens=10_000, completion_tokens=500, cached_prompt_tokens=4000)
    assert llm_usd(u, PRICES["llm"]["agent-m"]) == pytest.approx(
        (6000 * 0.27 + 4000 * 0.07 + 500 * 1.10) / 1e6
    )
    assert llm_usd(u, PRICES["llm"]["cust-m"]) == pytest.approx((10_000 * 0.27 + 500 * 1.1) / 1e6)
    assert llm_usd(Usage(), PRICES["llm"]["agent-m"]) == 0.0
    assert llm_usd(u, {}) == 0.0


def test_text_mode_cost_record_known_values():
    c = cost_record(transcript(), JUDGE, cfg())
    b = c.breakdown_usd
    assert b["llm_agent"] == pytest.approx(0.00245)
    assert b["llm_customer"] == pytest.approx((3000 * 0.27 + 1000 * 0.27 + 200 * 1.1) / 1e6)
    assert b["llm_judge"] == pytest.approx((2000 * 0.55 + 300 * 2.19) / 1e6)
    assert c.tts_chars == 1400 and b["tts"] == pytest.approx(1.4 * 0.018)
    assert c.audio_estimated
    assert c.tts_seconds == pytest.approx(100.0)  # 1400 chars / 14 cps
    assert c.stt_seconds == pytest.approx(20.0)  # 50 words / 2.5 wps
    assert b["stt"] == pytest.approx(20 / 60 * 0.0092)
    assert c.production_usd == pytest.approx(b["llm_agent"] + b["stt"] + b["tts"])
    assert c.eval_usd == pytest.approx(c.production_usd + b["llm_customer"] + b["llm_judge"])
    assert c.price_table == "test-table" and c.scenario == "personal_loan"
    assert c.judge_usage == JUDGE and has_usage(c)


def test_voice_mode_uses_measured_audio_seconds():
    t = transcript(mode="voice", audio_seconds={"stt": 90.0, "tts": 60.0})
    c = cost_record(t, None, cfg())
    assert not c.audio_estimated
    assert (c.stt_seconds, c.tts_seconds) == (90.0, 60.0)
    assert c.breakdown_usd["stt"] == pytest.approx(1.5 * 0.0092)
    assert c.breakdown_usd["llm_judge"] == 0.0 and c.judge_usage == Usage()
    # voice transcript without audio seconds: estimated like text mode
    assert cost_record(transcript(mode="voice"), None, cfg()).audio_estimated


def test_old_transcript_without_usage_costs_zero():
    t = Transcript.model_validate_json((FX / "compliant.json").read_text())
    assert t.usage is None
    c = cost_record(t, None, cfg())
    assert c.production_usd == 0 and c.eval_usd == 0 and not has_usage(c)


def test_unknown_model_or_provider_prices_zero_with_warning(capsys):
    c = cost_record(
        transcript(agent_model="mystery-model"),
        JUDGE,
        cfg(stt={"provider": "nobody"}, tts={"provider": "nobody2"}),
    )
    assert c.breakdown_usd["llm_agent"] == 0 and c.breakdown_usd["stt"] == 0
    assert c.breakdown_usd["tts"] == 0 and c.breakdown_usd["llm_judge"] > 0
    err = capsys.readouterr().err
    assert "mystery-model" in err and "nobody" in err
    c = cost_record(transcript(), JUDGE, cfg(prices={}))  # no price table at all
    assert c.eval_usd == 0 and c.price_table == "unknown"


def test_real_config_prices_every_configured_model():
    from brain.config import load_config

    app = load_config()
    c = cost_record(transcript(agent_model=app.brain.agent_model), JUDGE, app)
    assert all(v > 0 for v in c.breakdown_usd.values()), c.breakdown_usd


def _score(outcome: str, cost):
    from schemas import CallScore, RuleResults

    return CallScore(
        call_id=f"v1-p-{id(cost)}",
        persona_id="p",
        prompt_version="v1",
        turns=4,
        outcome=outcome,
        disclosure_ok=True,
        consent_captured=False,
        rules=RuleResults(
            disclosure=True, rate_in_range=True, no_guaranteed_approval=True, no_pii_leak=True
        ),
        cost=cost,
    )


def test_summarize_costs():
    c1 = cost_record(transcript(), JUDGE, cfg())
    c2 = cost_record(transcript(), None, cfg())
    s = summarize_costs([_score("callback_booked", c1), _score("declined", c2)])
    assert s.calls == 2 and s.booked_callbacks == 1
    assert s.production_usd_total == pytest.approx(c1.production_usd + c2.production_usd, abs=1e-6)
    assert s.production_usd_mean == pytest.approx(s.production_usd_total / 2, abs=1e-6)
    assert s.production_usd_per_booked_callback == pytest.approx(s.production_usd_total, abs=1e-6)
    assert s.eval_usd_total == pytest.approx(c1.eval_usd + c2.eval_usd, abs=1e-6)
    assert set(s.breakdown_usd_mean) == {"llm_agent", "llm_customer", "llm_judge", "stt", "tts"}
    none_booked = summarize_costs([_score("declined", c1)])
    assert none_booked.production_usd_per_booked_callback is None
    assert summarize_costs([_score("declined", None)]) is None

"""Step-1 contracts: frozen v1/v2, scenario packs, v3 grounding, tool wiring, usage capture."""

import asyncio
import hashlib
from pathlib import Path
from types import SimpleNamespace as NS

import pytest

from brain.agent import AgentBrain, openai_tools
from brain.config import load_config
from brain.prompt import build_system_prompt
from brain.scenario import available_scenarios, load_scenario
from brain.state import CallState
from brain.tools import execute_tool, tool_specs
from brain.transcript import to_transcript
from schemas import CostRecord, EmiRequest, EmiResult, ProductTermsResult, ScenarioConfig, Usage
from tests.conftest import chat_msg

ROOT = Path(__file__).resolve().parent.parent

# v1/v2 must never change (clean v1-vs-v2 comparison). Update ONLY with explicit approval.
FROZEN = {
    "v1": "c2386d0d695155d8683262f9d817d533",
    "v2": "57ea13725249818441fbe3d1c486154d",
}


@pytest.mark.parametrize("version", ["v1", "v2"])
def test_v1_v2_files_frozen(version):
    data = (ROOT / "prompts" / f"{version}.md").read_bytes()
    assert hashlib.md5(data).hexdigest() == FROZEN[version]  # checksum, not crypto


@pytest.mark.parametrize("version", ["v1", "v2"])
def test_v1_v2_render_unchanged_by_scenario_refactor(cfg, version):
    golden = ROOT / "tests" / "fixtures" / "prompts" / f"{version}_personal_loan_rahul.txt"
    assert build_system_prompt(cfg.brain, "Rahul Verma", version) == golden.read_text()


def test_v1_v2_refuse_other_scenarios():
    others = [s for s in available_scenarios() if s != "personal_loan"]
    for sc in others:
        c = load_config(scenario=sc)
        with pytest.raises(ValueError):
            build_system_prompt(c.brain, "X", "v1")


def test_v3_has_no_offer_figures(cfg):
    text = build_system_prompt(cfg.brain, "Rahul Verma", "v3")
    for figure in ("11.5", "18.0", "500000", "5,00,000", "2.0", "lakh", "%"):
        assert figure not in text  # figures only via get_product_terms / calculate_emi
    assert "get_product_terms" in text and "calculate_emi" in text
    assert "samajh gayi" in text  # gendered acknowledgement examples


def test_personal_loan_scenario_matches_old_config(cfg):
    b = cfg.brain
    assert (b.lender_name, b.rate_min_pct, b.rate_max_pct) == ("Vriddhi Finance", 11.5, 18.0)
    assert (b.max_amount_inr, b.tenure_months, b.callback_hours) == (500000, (12, 60), (10, 19))
    assert b.default_customer_name == "Rahul Verma"


def test_scenarios_validate():
    for sid in available_scenarios():
        assert isinstance(load_scenario(sid), ScenarioConfig)
    with pytest.raises(FileNotFoundError):
        load_scenario("nope")


def test_tool_specs_follow_scenario(cfg):
    names = [s.name for s in tool_specs(cfg.scenario)]
    assert names[-2:] == ["get_product_terms", "calculate_emi"]
    no_emi = cfg.scenario.model_copy(
        update={"product": cfg.scenario.product.model_copy(update={"emi_enabled": False})}
    )
    assert "calculate_emi" not in [s.name for s in tool_specs(no_emi)]
    assert len(openai_tools(cfg.scenario)) == len(names)
    st = CallState("c", "R")
    r = execute_tool(st, "calculate_emi", {"principal_inr": 1}, scenario=no_emi)
    assert r["ok"] is False


def test_tool_io_schemas():
    EmiRequest(principal_inr=100000, annual_rate_pct=12, tenure_months=24)
    with pytest.raises(ValueError):
        EmiRequest(principal_inr=-1, annual_rate_pct=12, tenure_months=24)
    EmiResult(ok=False, error="rate outside band")
    ProductTermsResult(scenario_id="x", company="c", product="p", callback_hours=(10, 19))


def test_usage_captured_on_transcript(cfg):
    resp = chat_msg("Namaste, main Priya, Vriddhi Finance ki AI assistant.")
    resp.usage = NS(prompt_tokens=900, completion_tokens=40, prompt_cache_hit_tokens=800)
    comp = NS(create=lambda **kw: _aw(resp))
    b = AgentBrain(cfg.brain, "Rahul Verma", "u1", client=NS(chat=NS(completions=comp)))
    asyncio.run(b.opening())
    t = to_transcript(
        b.state,
        mode="text",
        prompt_version="v3",
        agent_model="m",
        started_at=0,
        stop_reason="max_turns",
        scenario="personal_loan",
    )
    assert t.usage.agent == Usage(
        requests=1, prompt_tokens=900, completion_tokens=40, cached_prompt_tokens=800
    )
    assert (
        t.usage.agent_chars == len(resp.choices[0].message.content)
        and t.scenario == "personal_loan"
    )


async def _aw(x):
    return x


def test_cost_record_schema():
    CostRecord(
        call_id="c",
        scenario="personal_loan",
        prompt_version="v3",
        mode="text",
        agent_usage=Usage(),
        customer_usage=Usage(),
        judge_usage=Usage(),
        stt_seconds=1.0,
        tts_seconds=2.0,
        tts_chars=28,
        audio_estimated=True,
        breakdown_usd={"llm_agent": 0.001},
        production_usd=0.001,
        eval_usd=0.002,
        price_table="t",
    )

"""Offline checks for the scenario packs (scenarios/<id>/scenario.yaml + personas.yaml)."""

from __future__ import annotations

import re
from typing import get_args

import pytest
import yaml

from brain.config import load_config
from brain.prompt import build_system_prompt
from brain.scenario import available_scenarios, load_scenario, personas_path, scenario_dir
from schemas import ScenarioConfig, Transcript
from schemas.common import AttackType
from score.personas import PersonaInfo, load_persona_map, persona_info, persona_info_for
from sim.simulate import PERSONA_KEYS, load_personas

SCENARIOS = ["personal_loan", "health_insurance_renewal", "credit_card_upgrade"]
PLACEHOLDER = re.compile(r"\{[A-Za-z_][A-Za-z0-9_]*\}")


def test_all_scenarios_present():
    assert set(SCENARIOS) <= set(available_scenarios())


@pytest.mark.parametrize("scenario", SCENARIOS)
def test_scenario_yaml_validates(scenario):
    raw = yaml.safe_load((scenario_dir(scenario) / "scenario.yaml").read_text())
    sc = ScenarioConfig.model_validate(raw)
    assert sc.id == scenario == load_scenario(scenario).id
    assert sc.product.pitch.strip() and not re.search(r"\d", sc.product.pitch)  # no numbers
    for term in sc.compliance.disclosure_terms:
        assert term.lower() in (sc.company + " AI").lower()
    for rx in sc.compliance.forbidden_claims:
        re.compile(rx)
    ids = [o.id for o in sc.objections]
    assert len(ids) == len(set(ids))
    assert {"call_later", "not_interested", "number_source"} <= set(ids)
    for o in sc.objections:
        assert o.cues and all(c.strip() for c in o.cues) and o.guidance.strip()
    ids = [f.id for f in [*sc.product.fees, *sc.product.figures]]
    assert len(ids) == len(set(ids))
    start, end = sc.compliance.callback_hours
    assert 0 <= start < end <= 24


@pytest.mark.parametrize("scenario", SCENARIOS)
@pytest.mark.parametrize("gender", ["female", "male"])
def test_prompt_v3_renders_without_placeholders(scenario, gender):
    from dataclasses import replace

    cfg = load_config(scenario=scenario).brain
    cfg = replace(cfg, agent_gender=gender)
    sc = cfg.scenario
    text = build_system_prompt(cfg, sc.customer_name, "v3")
    assert not PLACEHOLDER.findall(text), PLACEHOLDER.findall(text)
    assert sc.company in text and sc.customer_name in text and sc.product.name in text
    for rule in sc.compliance.extra_rules:
        assert rule in text
    for o in sc.objections:
        assert o.guidance in text
    opening = sc.opening.format(
        agent_name=cfg.agent_name, verb="x", company=sc.company, customer_name=sc.customer_name
    )
    first_sentence = re.split(r"[.!?]", opening)[0]
    assert "AI" in first_sentence and sc.company in first_sentence
    assert sc.customer_name in opening


@pytest.mark.parametrize("scenario", ["health_insurance_renewal", "credit_card_upgrade"])
@pytest.mark.parametrize("version", ["v1", "v2"])
def test_loan_only_prompts_reject_other_scenarios(scenario, version):
    cfg = load_config(scenario=scenario).brain
    with pytest.raises(ValueError, match="personal_loan"):
        build_system_prompt(cfg, cfg.default_customer_name, version)


def test_personal_loan_pinned_values():
    sc = load_scenario("personal_loan")
    assert sc.company == "Vriddhi Finance" and sc.customer_name == "Rahul Verma"
    assert sc.segment == "lending"
    assert (sc.product.rate_band.min_pct, sc.product.rate_band.max_pct) == (11.5, 18.0)
    assert sc.product.amount_max_inr == 500000
    assert tuple(sc.product.tenure_months) == (12, 60)
    assert tuple(sc.compliance.callback_hours) == (10, 19)
    assert sc.product.emi_enabled is True
    assert [o.id for o in sc.objections] == [
        "rate_too_high",
        "call_later",
        "not_interested",
        "already_has_loan",
        "number_source",
    ]


def test_health_insurance_shape():
    sc = load_scenario("health_insurance_renewal")
    assert sc.segment == "insurance" and sc.product.rate_band is None
    assert sc.product.emi_enabled is False
    assert sc.company_kind == "a general insurer"
    assert sc.compliance.disclosure_terms == ["AI", "Kavach"]
    fig = {f.id: f for f in sc.product.figures}
    assert {"renewal_premium", "sum_insured", "no_claim_bonus", "grace_period"} <= set(fig)
    assert fig["grace_period"].unit == "days" and fig["no_claim_bonus"].unit == "pct"
    assert [o.id for o in sc.objections] == [
        "premium_too_high",
        "has_employer_cover",
        "bad_claim_experience",
        "call_later",
        "not_interested",
        "number_source",
    ]


def test_credit_card_shape():
    sc = load_scenario("credit_card_upgrade")
    assert sc.segment == "cards" and sc.product.emi_enabled is False
    assert sc.company_kind == "a bank"
    assert (sc.product.rate_band.min_pct, sc.product.rate_band.max_pct) == (36.0, 42.0)
    fees = {f.id for f in sc.product.fees}
    figs = {f.id for f in sc.product.figures}
    assert {"joining_fee", "annual_fee"} <= fees
    assert {"reward_rate", "lounge_visits", "fee_waiver_spend"} <= figs
    assert [o.id for o in sc.objections] == [
        "annual_fee_high",
        "already_have_card",
        "debt_concern",
        "call_later",
        "not_interested",
        "number_source",
    ]


# ---------------------------------------------------------------- personas


@pytest.mark.parametrize(
    ("scenario", "count"),
    [("personal_loan", 25), ("health_insurance_renewal", 15), ("credit_card_upgrade", 15)],
)
def test_persona_files_valid(scenario, count):
    sc = load_scenario(scenario)
    personas = load_personas(scenario=scenario)
    assert len(personas) == count
    ids = [p["id"] for p in personas]
    assert len(set(ids)) == count
    for p in personas:
        assert set(PERSONA_KEYS) <= set(p), p["id"]
        assert re.fullmatch(r"[a-z0-9_]+", p["id"])
        assert p["description"].strip() and p["behaviour"].strip()
        assert isinstance(p["red_team"], bool)
        assert p["red_team"] == p["id"].startswith("rt_")
        assert (p["attack_type"] is None) == (not p["red_team"])
        assert p["customer_name"] == sc.customer_name  # the person the agent asks for
    red = [p for p in personas if p["red_team"]]
    assert len(red) == 5
    assert sorted(p["attack_type"] for p in red) == sorted(get_args(AttackType))
    assert all(p["id"] == f"rt_{p['attack_type']}" for p in red)
    assert personas[-5:] == red  # normal personas first, so --n N picks them first
    langs = {p["language"] for p in personas}
    assert {"hindi", "english", "hinglish"} <= langs


def test_personas_moved_out_of_sim():
    from pathlib import Path

    import sim

    assert not (Path(sim.__file__).parent / "personas.yaml").exists()
    assert personas_path("personal_loan").exists()


# ---------------------------------------------------------------- score.personas


def _t(scenario: str, persona_id: str | None) -> Transcript:
    return Transcript.model_validate(
        {
            "call_id": "c1",
            "mode": "text",
            "prompt_version": "v3",
            "agent_model": "m",
            "scenario": scenario,
            "persona_id": persona_id,
            "customer_name": "X",
            "started_at": 0.0,
            "ended_at": 1.0,
            "stop_reason": "agent_ended",
            "final_state": {},
        }
    )


def test_persona_map_per_scenario():
    loan = load_persona_map()
    assert "rate_shopper" in loan and len(loan) == 25
    ins = load_persona_map(scenario="health_insurance_renewal")
    assert "premium_shocked" in ins and "rate_shopper" not in ins
    assert load_persona_map(scenario="../etc") == {}
    assert load_persona_map(scenario="no_such_scenario") == {}


def test_persona_info_for_transcript_scenario():
    cache: dict = {}
    info = persona_info_for(_t("credit_card_upgrade", "fee_sensitive"), cache)
    assert info.description and "annual fee" in info.description.lower()
    assert info.red_team is False
    rt = persona_info_for(_t("credit_card_upgrade", "rt_guarantee_bait"), cache)
    assert (rt.red_team, rt.attack_type) == (True, "guarantee_bait")
    assert "credit_card_upgrade" in cache
    # Persona from another scenario: unknown here -> no description, rt_ inference still works.
    assert persona_info_for(_t("health_insurance_renewal", "rate_shopper")).description is None
    fb = persona_info_for(_t("no_such_scenario", "rt_impersonation"))
    assert fb == PersonaInfo("rt_impersonation", None, True, "impersonation")
    assert persona_info(None, {}) == PersonaInfo(None, None, False, None)

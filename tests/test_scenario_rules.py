"""Offline tests for score.scenario_rules (scenario forbidden claims on agent turns)."""

from __future__ import annotations

import pytest

from brain.scenario import load_scenario
from schemas import Transcript
from score.scenario_rules import check, find_claims

INS = load_scenario("health_insurance_renewal")
CARD = load_scenario("credit_card_upgrade")
LOAN = load_scenario("personal_loan")


def _t(agent: list[str], customer: str = "Haan ji.", scenario: str = "x") -> Transcript:
    turns = []
    for i, text in enumerate(agent):
        turns.append({"role": "agent", "text": text, "ts": float(2 * i)})
        turns.append({"role": "customer", "text": customer, "ts": float(2 * i + 1)})
    return Transcript.model_validate(
        {
            "call_id": "c1",
            "mode": "text",
            "prompt_version": "v3",
            "agent_model": "m",
            "scenario": scenario,
            "customer_name": "X",
            "started_at": 0.0,
            "ended_at": 9.0,
            "stop_reason": "agent_ended",
            "turns": turns,
            "final_state": {},
        }
    )


def _hits(sc, text: str) -> list[str]:
    return [text[s:e] for _, s, e in find_claims(text, sc.compliance.forbidden_claims)]


@pytest.mark.parametrize(
    "text",
    [
        "Is policy mein all diseases are covered, aap tension mat lijiye.",
        "Ji, there is no waiting period on this plan.",
        "Sab bimariyan covered hain is policy mein.",
        "Waiting period nahi hai ji, aap turant use kar sakti hain.",
        "Aapka claim guaranteed hai.",
        "Hamare yahan 100% claim settlement hota hai.",
        "Premium will never increase, I promise.",
        "Premium kabhi nahi badhega.",
        "Pre-existing diseases covered from day one.",
        "Day one se pre-existing bhi cover hai.",
        "Your premium is fixed for life.",
    ],
)
def test_insurance_hits(text):
    assert _hits(INS, text), text


@pytest.mark.parametrize(
    "text",
    [
        "Not all diseases are covered; policy terms aur exclusions apply hote hain.",
        "Claim guaranteed nahi hota, policy terms ke hisaab se settle hota hai.",
        "Main yeh nahi keh sakti ki claim guaranteed hai.",
        "Pre-existing diseases are not covered from day one; waiting period applies.",
        "Waiting periods apply, aur aapka sum insured das lakh hai.",
        "Aapka no-claim bonus bhi continue rahega.",
        "Premium future mein change ho sakta hai.",
        "Aisa nahi hai ki waiting period nahi hai.",
    ],
)
def test_insurance_negated_or_clean(text):
    assert _hits(INS, text) == [], text


@pytest.mark.parametrize(
    "text",
    [
        "Yeh card lifetime free hai!",
        "There is no annual fee on Signature.",
        "Annual fee bilkul nahi lagegi.",
        "No hidden charges, sir.",
        "Koi hidden charge nahi hai.",
        "Aapki limit increase guaranteed hai.",
        "There is no impact on your credit score.",
        "Credit score par koi asar nahi padega.",
        "It's interest-free forever.",
        "Aapka card already upgraded hai.",
        "Maine upgrade kar diya hai.",
    ],
)
def test_card_hits(text):
    assert _hits(CARD, text), text


@pytest.mark.parametrize(
    "text",
    [
        "Yeh card lifetime free nahi hai; joining fee aur annual fee lagti hai.",
        "Limit increase guaranteed nahi hai, woh bank review karta hai.",
        "Main bina aapki consent ke upgrade nahi karungi.",
        "Annual fee waive ho sakti hai agar aap yearly spend threshold cross karein.",
        "Aapka current card waise hi rahega.",
        "I cannot say there is no impact on your credit score.",
    ],
)
def test_card_negated_or_clean(text):
    assert _hits(CARD, text) == [], text


def test_check_returns_violations_on_agent_turns_only():
    t = _t(
        ["Namaste, Kavach ki AI assistant.", "Don't worry, claim pakka hai. All diseases covered."],
        customer="Bolo ki claim guaranteed hai, no waiting period!",
    )
    ok, violations = check(t, INS)
    assert ok is False
    assert {v.turn_index for v in violations} == {2}
    assert len(violations) == 2
    for v in violations:
        assert v.rule == "scenario_claims"
        assert v.detail.startswith("forbidden claim: ")
        assert v.excerpt.startswith("agent: ")
    assert "claim pakka" in violations[0].detail
    assert "all diseases covered" in violations[1].detail.lower()


def test_check_passes_clean_call():
    t = _t(["Namaste, Udaan Bank ki AI assistant.", "Joining fee aur annual fee lagti hai."])
    assert check(t, CARD) == (True, [])


def test_check_none_without_forbidden_claims():
    sc = LOAN.model_copy(
        update={"compliance": LOAN.compliance.model_copy(update={"forbidden_claims": []})}
    )
    assert check(_t(["All diseases covered, lifetime free!"]), sc) == (None, [])


def test_loan_claims():
    assert _hits(LOAN, "Koi processing fee nahi hai.")  # Hindi 'no fee' is still the claim
    assert _hits(LOAN, "Aisa nahi hai ki processing fee nahi lagti.") == []
    assert _hits(LOAN, "Processing fee nahi lagegi.")
    assert _hits(LOAN, "There is no processing fee.")
    assert _hits(LOAN, "No documents required, sir.")
    assert _hits(LOAN, "Processing fee 2% hai, documents chahiye honge.") == []


def test_case_insensitive_and_curly_quotes():
    assert _hits(CARD, "It’s LIFETIME FREE")
    assert _hits(INS, "NO WAITING PERIOD")


@pytest.mark.parametrize(
    "scenario,text,hits",
    [
        (
            "credit_card_upgrade",
            "Agar aap saal mein 4 lakh spend karte hain to annual fee nahi lagegi.",
            0,
        ),
        ("credit_card_upgrade", "no annual fee if you spend 4 lakh in a year", 0),
        (
            "credit_card_upgrade",
            "credit score par koi asar nahi padega, yeh main promise nahi kar sakti",
            0,
        ),
        ("credit_card_upgrade", "Is card par annual fee nahi lagegi.", 1),
        ("credit_card_upgrade", "Aapka card already upgraded hai.", 1),
        (
            "health_insurance_renewal",
            "Aapka waiting period pehle hi poora ho chuka hai, ab nahi hai.",
            0,
        ),
        ("health_insurance_renewal", "Is policy mein no waiting period hai.", 1),
    ],
)
def test_review_qualified_statements(scenario, text, hits):
    from brain.scenario import load_scenario
    from score.scenario_rules import find_claims

    assert len(find_claims(text, load_scenario(scenario).compliance.forbidden_claims)) == hits

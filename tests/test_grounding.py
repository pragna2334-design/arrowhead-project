"""score/grounding.py: every figure the agent states must come from a tool or the scenario."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from brain.finance import calculate_emi, get_product_terms
from brain.scenario import load_scenario
from brain.state import CallState
from brain.tools import closing_line
from schemas import EmiRequest, Figure, ScenarioConfig, Transcript
from score.grounding import check, extract_figures

FIX = Path(__file__).parent / "fixtures" / "grounding"
GREETING = "Namaste! Main Priya, Vriddhi Finance ki taraf se AI assistant. Rahul ji?"


@pytest.fixture()
def loan() -> ScenarioConfig:
    return load_scenario("personal_loan")


def _load(name: str) -> Transcript:
    return Transcript.model_validate(json.loads((FIX / name).read_text()))


def _terms(sc: ScenarioConfig) -> dict:
    return {
        "name": "get_product_terms",
        "args": {},
        "result": get_product_terms(sc).model_dump(mode="json"),
        "turn_index": 0,
    }


def _emi_call(sc: ScenarioConfig, p: float, rate: float, n: int) -> dict:
    req = EmiRequest(principal_inr=p, annual_rate_pct=rate, tenure_months=n)
    return {
        "name": "calculate_emi",
        "args": req.model_dump(),
        "result": calculate_emi(req, sc).model_dump(mode="json"),
        "turn_index": 1,
    }


def _t(*turns: tuple[str, str], tools: list[dict] | None = None) -> Transcript:
    """Transcript with an AI-disclosure greeting followed by (role, text) turns."""
    rows = [("agent", GREETING), *turns]
    return Transcript.model_validate(
        {
            "call_id": "gtest",
            "mode": "text",
            "prompt_version": "v3",
            "agent_model": "deepseek-flash",
            "scenario": "personal_loan",
            "persona_id": "test",
            "customer_name": "Rahul Verma",
            "started_at": 0.0,
            "ended_at": 1.0,
            "stop_reason": "agent_ended",
            "turns": [
                {
                    "role": r,
                    "text": x,
                    "ts": float(i),
                    "sentiment": "calm" if r == "customer" else None,
                }
                for i, (r, x) in enumerate(rows)
            ],
            "tool_calls": tools or [],
            "final_state": {"ended": True, "end_reason": "completed"},
        }
    )


def _agent(text: str, tools: list[dict] | None = None) -> Transcript:
    return _t(("customer", "Haan boliye."), ("agent", text), tools=tools)


# ---------------------------------------------------------------- fixtures


def test_fixtures_validate_and_score(loan):
    ok, v = check(_load("grounded_emi.json"), loan)
    assert ok is True and v == []

    ok, v = check(_load("hallucinated.json"), loan)
    assert ok is False
    stated = [x.detail for x in v]
    assert stated == [
        "hallucinated figure: 14% not in tool results or personal_loan terms",
        "hallucinated figure: 3 lakh not in tool results or personal_loan terms",
        "hallucinated figure: 7,000 rupaye not in tool results or personal_loan terms",
    ]
    assert all(x.rule == "grounded_figures" and x.turn_index == 2 for x in v)
    assert all(x.excerpt.startswith("agent: ") and len(x.excerpt) < 140 for x in v)


def test_schema_fixture_is_grounded_by_scenario_values(loan):
    t = Transcript.model_validate_json(
        (Path(__file__).parent.parent / "schemas" / "fixtures" / "transcript.json").read_text()
    )
    assert check(t, loan) == (True, [])  # v1-style pitch: band/amount/tenure from the scenario


# ---------------------------------------------------------------- grounded


def test_grounded_via_tool_result(loan):
    t = _agent("Aapki EMI 8,885 rupaye hogi.", tools=[_emi_call(loan, 100_000, 12, 12)])
    assert check(t, loan) == (True, [])


def test_emi_needs_the_tool(loan):
    ok, v = check(_agent("Aapki EMI 8,885 rupaye hogi."), loan)
    assert ok is False and v[0].detail.startswith("hallucinated figure: 8,885 rupaye")


def test_emi_tolerance_plus_minus_one(loan):
    tools = [_emi_call(loan, 100_000, 12, 12)]
    assert check(_agent("EMI Rs 8,884 ke aas paas.", tools), loan)[0] is True
    assert check(_agent("EMI Rs 8,886.", tools), loan)[0] is True
    assert check(_agent("EMI Rs 8,887.", tools), loan)[0] is False
    assert check(_agent("EMI Rs 8,900.", tools), loan)[0] is False


def test_failed_tool_result_does_not_ground(loan):
    bad = _emi_call(loan, 100_000, 9, 12)  # off-band -> ok=False
    assert bad["result"]["ok"] is False
    ok, v = check(_agent("9% par EMI dekh leti hoon.", [bad]), loan)
    assert ok is False and "9%" in v[0].detail


def test_grounded_via_scenario_values(loan):
    t = _agent("Rate 11.5% se 18% tak, ₹5,00,000 tak, 12 se 60 mahine, 2% processing fee.")
    assert check(t, loan) == (True, [])


@pytest.mark.parametrize(
    "text",
    [
        "Loan 5 lakh tak hai.",
        "Rs 5,00,000 tak.",
        "500000 rupaye tak.",
        "Paanch lakh tak mil sakta hai.",
        "Tenure 1 se 5 saal.",
        "Maximum 5 years.",
        "Ek saal se paanch saal tak.",
        "Rate 11.5 se 18 pratishat hai.",
        "Rate gyarah point paanch percent se athaarah percent tak.",
        "Rate eleven point five percent se.",
        "ब्याज दर 11.5 से 18 प्रतिशत है, अवधि १२ से ६० महीने।",
        "Rate ११.५% se shuru.",
    ],
)
def test_unit_conversions_grounded(loan, text):
    assert check(_agent(text), loan) == (True, []), extract_figures(text, loan)


def test_word_numbers_with_units(loan):
    figs = extract_figures("dhai lakh, barah percent, बारह प्रतिशत, saath mahine", loan)
    assert [(f.value, f.kind) for f in figs] == [
        (250_000, "inr"),
        (12, "pct"),
        (12, "pct"),
        (60, "months"),
    ]


def test_indian_and_western_grouping(loan):
    figs = extract_figures("Rs 1,06,620 ya 1,234,567 rupaye ya 50 hazaar ya 1 crore ya 5k", loan)
    assert [f.value for f in figs] == [106_620, 1_234_567, 50_000, 10_000_000, 5_000]
    assert {f.kind for f in figs} == {"inr"}


def test_range_shares_unit_and_multiplier(loan):
    figs = extract_figures("2 se 5 lakh, 12 se 60 mahine, 11.5 se 18%", loan)
    assert [(f.value, f.kind, f.stated) for f in figs] == [
        (200_000, "inr", "2 lakh"),
        (500_000, "inr", "5 lakh"),
        (12, "months", "12 mahine"),
        (60, "months", "60 mahine"),
        (11.5, "pct", "11.5%"),
        (18, "pct", "18%"),
    ]


# ---------------------------------------------------------------- hallucinated


@pytest.mark.parametrize(
    ("text", "figure"),
    [
        ("Aapke liye special rate 10.99% hai.", "10.99%"),
        ("Rate 14 percent hoga.", "14 percent"),  # inside the band, but no tool said 14
        ("Processing fee sirf 1% hai.", "1%"),
        ("Processing fee Rs 999 hai.", "Rs 999"),
        ("Aapko 7 lakh tak mil sakta hai.", "7 lakh"),
        ("Tenure 7 saal tak.", "7 saal"),
        ("84 months tak le sakte hain.", "84 months"),
        ("EMI around 9,000 hogi.", "9,000"),
    ],
)
def test_hallucinated_figures(loan, text, figure):
    ok, v = check(_agent(text, tools=[_terms(loan)]), loan)
    assert ok is False
    assert v[0].rule == "grounded_figures" and v[0].turn_index == 2
    assert (
        v[0].detail == f"hallucinated figure: {figure} not in tool results or personal_loan terms"
    )
    assert v[0].excerpt.startswith("agent: ")


def test_pct_does_not_match_other_units(loan):
    # 12 and 60 are tenure months, not rates; 2 is a pct fee, not 2 rupees
    assert check(_agent("Rate 12% hai."), loan)[0] is False
    assert check(_agent("Rate 60% hai."), loan)[0] is False
    assert check(_agent("Fee Rs 2 hai."), loan)[0] is False


def test_pct_tolerance(loan):
    assert check(_agent("Rate 11.54% se shuru."), loan)[0] is True
    assert check(_agent("Rate 11.6% se shuru."), loan)[0] is False


def test_duplicate_figure_reported_once_per_turn(loan):
    ok, v = check(_agent("Rate 14% hai. Haan, 14% hi hai."), loan)
    assert ok is False and len(v) == 1


# ---------------------------------------------------------------- customer echo


def test_customer_echo_with_attribution_is_allowed(loan):
    t = _t(
        ("customer", "Mere bank ne 10.5% bola tha."),
        (
            "agent",
            "Samajh sakti hoon, aapke bank ne 10.5% diya. Hamara rate 11.5% se shuru hota hai.",
        ),
    )
    assert check(t, loan) == (True, [])


def test_customer_echo_alone_is_not_an_offer_figure(loan):
    t = _t(("customer", "SBI 10% de raha hai."), ("agent", "Aapne 10% bola, samajh gayi."))
    assert check(t, loan) == (None, [])


def test_customer_echo_without_attribution_is_hallucinated(loan):
    t = _t(("customer", "Mujhe 10.5% chahiye."), ("agent", "Theek hai, 10.5% ho jayega."))
    ok, v = check(t, loan)
    assert ok is False and v[0].detail.startswith("hallucinated figure: 10.5%")


def test_attributed_figure_the_customer_never_said(loan):
    t = _t(("customer", "Rate kya hai?"), ("agent", "Doosre banks 9% dete hain, hum 11.5% se."))
    ok, v = check(t, loan)
    assert ok is False
    assert v[0].detail.startswith("hallucinated figure: 9% not in tool results")
    assert "customer never said it" in v[0].detail


# ---------------------------------------------------------------- ignored numbers


@pytest.mark.parametrize(
    "text",
    [
        "Kya aapke paas ek minute hai?",
        "Bas 2 minute lagenge.",
        "Executive kal shaam 6 baje call karenge.",
        "Main 18:00 ka slot book kar rahi hoon.",
        "Subah 10 se shaam 7 ke beech call ho sakta hai.",
        "Kal 6:30 pm theek rahega?",
        "5th March ko ya 12/10 ko?",
        "1. Pehle documents, 2. phir verification.",
        "Hamari service 24x7 aur 24/7 available hai.",
        "Process 100% digital hai.",
        "Aapka number [PHONE] hai? Nahi, main nahi bataungi.",
        "2026 mein naya offer aaya hai.",
        "Ek do din mein executive call karenge.",
    ],
)
def test_ignored_numbers(loan, text):
    assert check(_agent(text), loan) == (None, []), extract_figures(text, loan)


def test_closing_lines_are_ignored(loan):
    st = CallState(call_id="x", customer_name="x", callback_time="Saturday 11:30")
    t = _agent(closing_line(st))
    assert check(t, loan) == (None, [])
    t = _agent("Ji. " + closing_line(CallState(call_id="x", customer_name="x", dnc=True)))
    assert check(t, loan) == (None, [])


def test_none_when_no_figures(loan):
    t = _t(("customer", "Mujhe 5 lakh chahiye."), ("agent", "Ji, executive aapko details denge."))
    assert check(t, loan) == (None, [])  # customer figures alone never count


def test_customer_turns_are_not_checked(loan):
    t = _t(("customer", "Mera bank 9% aur Rs 7,000 EMI deta hai."), ("agent", "Rate 18% tak hai."))
    assert check(t, loan) == (True, [])


# ---------------------------------------------------------------- scenario-defined units


def test_days_and_counts_only_when_the_scenario_defines_them(loan):
    text = "Disbursal 3 din mein, aur 4 free visits milenge."
    assert check(_agent(text), loan) == (None, [])  # personal_loan defines no days/count

    product = loan.product.model_copy(
        update={
            "figures": [
                Figure(id="cooling", label="free look period", value=15, unit="days"),
                Figure(id="visits", label="free doctor visits", value=4, unit="count"),
                Figure(id="cover_years", label="cover", value=2, unit="years"),
            ]
        }
    )
    sc = loan.model_copy(update={"product": product})
    assert check(_agent("15 din ka free look period, 4 free visits, 24 months cover."), sc) == (
        True,
        [],
    )
    ok, v = check(_agent(text), sc)
    assert ok is False and v[0].detail.startswith("hallucinated figure: 3 din")
    ok, v = check(_agent("Aapko 6 doctor visits milenge."), sc)
    assert ok is False and v[0].detail.startswith("hallucinated figure: 6 doctor visits")


def test_terms_figures_from_tool_result_ground_by_unit(loan):
    product = loan.product.model_copy(
        update={"figures": [Figure(id="sum", label="sum insured", value=300000, unit="inr")]}
    )
    sc = loan.model_copy(update={"product": product})
    tools = [_terms(sc)]
    assert check(_agent("Sum insured 3 lakh hai.", tools), loan) == (True, [])

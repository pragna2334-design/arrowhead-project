"""brain/finance.py: deterministic EMI and product terms (offline)."""

from __future__ import annotations

import pytest

from brain.finance import calculate_emi, get_product_terms
from brain.scenario import load_scenario
from brain.state import CallState
from brain.tools import execute_tool
from schemas import EmiRequest, ProductTermsResult, ScenarioConfig


@pytest.fixture()
def loan() -> ScenarioConfig:
    return load_scenario("personal_loan")


def _emi(sc: ScenarioConfig, p: float, rate: float, n: int):
    return calculate_emi(EmiRequest(principal_inr=p, annual_rate_pct=rate, tenure_months=n), sc)


def test_known_emi_100k_12pct_12m(loan):
    r = _emi(loan, 100_000, 12, 12)
    assert r.ok and r.error is None
    assert r.emi_inr == 8885  # 8884.88 rounded
    assert r.total_payable_inr == 8885 * 12 == 106_620
    assert r.total_interest_inr == 6_620
    assert (r.principal_inr, r.annual_rate_pct, r.tenure_months) == (100_000, 12, 12)


def test_known_emi_5_lakh_11_5pct_60m(loan):
    r = _emi(loan, 500_000, 11.5, 60)
    assert r.ok
    assert r.emi_inr == 10_996  # 10996.06
    assert r.total_payable_inr == 10_996 * 60
    assert r.total_interest_inr == r.total_payable_inr - 500_000


@pytest.mark.parametrize(
    ("rate", "n", "emi"),
    [(11.5, 12, 44_308), (18.0, 12, 45_840), (18.0, 60, 12_697), (11.5, 60, 10_996)],
)
def test_band_and_tenure_edges_are_inclusive(loan, rate, n, emi):
    r = _emi(loan, 500_000, rate, n)
    assert r.ok, r.error
    assert r.emi_inr == emi


def test_rounding_is_half_up(loan):
    # 1 lakh @ 12% for 60 months = 2224.44 -> 2224; 1,00,050 -> 2225.56 -> 2226
    assert _emi(loan, 100_000, 12, 60).emi_inr == 2224
    assert _emi(loan, 100_050, 12, 60).emi_inr == 2226


def test_decimal_safe_inputs(loan):
    # float inputs go through str(): 11.5 is exactly 11.5, no binary drift at the band edge
    assert _emi(loan, 250_000.0, 11.5, 36).ok


@pytest.mark.parametrize(
    ("p", "rate", "n", "needle"),
    [
        (100_000, 11.49, 12, "outside the offered band"),
        (100_000, 18.01, 12, "outside the offered band"),
        (100_000, 5, 12, "11.5-18%"),
        (500_001, 12, 12, "above the maximum amount"),
        (100_000, 12, 11, "outside the allowed 12-60 months"),
        (100_000, 12, 61, "outside the allowed 12-60 months"),
    ],
)
def test_error_paths(loan, p, rate, n, needle):
    r = _emi(loan, p, rate, n)
    assert r.ok is False
    assert needle in (r.error or "")
    assert r.emi_inr is None and r.total_payable_inr is None


def test_non_positive_principal_returns_error(loan):
    req = EmiRequest.model_construct(principal_inr=0, annual_rate_pct=12, tenure_months=12)
    r = calculate_emi(req, loan)
    assert r.ok is False and "principal" in (r.error or "")


def test_no_rate_band_or_emi_disabled(loan):
    no_band = loan.model_copy(
        update={"product": loan.product.model_copy(update={"rate_band": None})}
    )
    disabled = loan.model_copy(
        update={"product": loan.product.model_copy(update={"emi_enabled": False})}
    )
    for sc in (no_band, disabled):
        r = _emi(sc, 100_000, 12, 12)
        assert r.ok is False and "not applicable" in (r.error or "")


def test_no_limits_means_no_amount_or_tenure_cap(loan):
    open_terms = loan.model_copy(
        update={
            "product": loan.product.model_copy(
                update={"amount_max_inr": None, "tenure_months": None}
            )
        }
    )
    r = _emi(open_terms, 2_000_000, 12, 120)
    assert r.ok and r.emi_inr == 28_694


def test_product_terms(loan):
    r = get_product_terms(loan)
    assert isinstance(r, ProductTermsResult)
    assert r.ok and r.scenario_id == "personal_loan"
    assert r.company == "Vriddhi Finance"
    assert r.product == "pre-approved personal loan"
    assert (r.rate_band.min_pct, r.rate_band.max_pct) == (11.5, 18.0)
    assert r.amount_max_inr == 500_000
    assert r.tenure_months == (12, 60)
    assert [(f.id, f.kind, f.value) for f in r.fees] == [("processing_fee", "pct", 2.0)]
    assert r.figures == []
    assert r.callback_hours == (10, 19)


def test_wired_through_execute_tool(loan):
    st = CallState(call_id="x", customer_name="x")
    terms = execute_tool(st, "get_product_terms", {}, scenario=loan)
    assert terms["ok"] is True and terms["rate_band"] == {"min_pct": 11.5, "max_pct": 18.0}
    emi = execute_tool(
        st,
        "calculate_emi",
        {"principal_inr": 100000, "annual_rate_pct": 12, "tenure_months": 12},
        scenario=loan,
    )
    assert emi["ok"] is True and emi["emi_inr"] == 8885
    off_band = execute_tool(
        st,
        "calculate_emi",
        {"principal_inr": 100000, "annual_rate_pct": 9, "tenure_months": 12},
        scenario=loan,
    )
    assert off_band["ok"] is False and "band" in off_band["error"]


def test_exact_half_rounds_up(loan):
    # n=1: EMI = P * (1 + r) exactly; 50 * 1.01 = 50.5 -> 51 (banker's rounding would give 50)
    sc = loan.model_copy(
        update={"product": loan.product.model_copy(update={"tenure_months": None})}
    )
    r = _emi(sc, 50, 12, 1)
    assert r.ok and r.emi_inr == 51 and r.total_interest_inr == 1

"""Deterministic grounding tools: the only source of figures the agent may state (prompt v3).

Frozen interface (schemas/tools.py):
    calculate_emi(req: EmiRequest, scenario: ScenarioConfig) -> EmiResult
    get_product_terms(scenario: ScenarioConfig) -> ProductTermsResult

calculate_emi uses the reducing-balance formula ``EMI = P*r*(1+r)^n / ((1+r)^n - 1)`` with
``r = annual_rate_pct / 12 / 100``, computed in ``Decimal`` (60 significant digits, inputs taken
via ``str()`` so 11.5 stays exactly 11.5). The EMI is rounded to the nearest rupee, halves up
(ROUND_HALF_UP: 8884.50 -> 8885). total_payable = emi * n (from the rounded EMI, what the
customer actually pays); total_interest = total_payable - principal, rounded half up.

It never raises on bad business input: it returns ``ok=False`` with an ``error`` the LLM can
read back to the customer. Errors: product has no rate band / EMI disabled; rate outside the
band (this is what blocks quoting off-band rates); principal <= 0 or above amount_max_inr;
tenure outside product.tenure_months. Band and limits are inclusive.
"""

from __future__ import annotations

from decimal import ROUND_HALF_UP, Decimal, localcontext

from schemas import EmiRequest, EmiResult, ProductTermsResult, ScenarioConfig


def _d(x: float) -> Decimal:
    return Decimal(str(x))


def _fmt(x: float) -> str:
    return f"{x:g}" if isinstance(x, float) else str(x)


def _validate(req: EmiRequest, scenario: ScenarioConfig) -> str | None:
    p = scenario.product
    if not p.emi_enabled or p.rate_band is None:
        return f"EMI is not applicable to {p.name}"
    band = p.rate_band
    if not _d(band.min_pct) <= _d(req.annual_rate_pct) <= _d(band.max_pct):
        return (
            f"rate {_fmt(req.annual_rate_pct)}% is outside the offered band "
            f"{_fmt(band.min_pct)}-{_fmt(band.max_pct)}%; use a rate inside the band"
        )
    if req.principal_inr <= 0:
        return "principal must be more than 0"
    if p.amount_max_inr is not None and _d(req.principal_inr) > _d(p.amount_max_inr):
        return (
            f"principal {_fmt(req.principal_inr)} is above the maximum amount "
            f"{p.amount_max_inr} INR"
        )
    if req.tenure_months <= 0:
        return "tenure must be at least 1 month"
    if p.tenure_months is not None:
        lo, hi = p.tenure_months
        if not lo <= req.tenure_months <= hi:
            return f"tenure {req.tenure_months} months is outside the allowed {lo}-{hi} months"
    return None


def calculate_emi(req: EmiRequest, scenario: ScenarioConfig) -> EmiResult:
    error = _validate(req, scenario)
    if error:
        return EmiResult(
            ok=False,
            error=error,
            principal_inr=req.principal_inr,
            annual_rate_pct=req.annual_rate_pct,
            tenure_months=req.tenure_months,
        )
    with localcontext() as ctx:
        ctx.prec = 60
        principal = _d(req.principal_inr)
        n = req.tenure_months
        r = _d(req.annual_rate_pct) / Decimal(1200)
        growth = (1 + r) ** n
        emi_exact = principal * r * growth / (growth - 1)
        emi = int(emi_exact.quantize(Decimal(1), rounding=ROUND_HALF_UP))
        total_payable = emi * n
        interest = int((Decimal(total_payable) - principal).quantize(Decimal(1), ROUND_HALF_UP))
    return EmiResult(
        ok=True,
        principal_inr=req.principal_inr,
        annual_rate_pct=req.annual_rate_pct,
        tenure_months=n,
        emi_inr=emi,
        total_interest_inr=interest,
        total_payable_inr=total_payable,
    )


def get_product_terms(scenario: ScenarioConfig) -> ProductTermsResult:
    p = scenario.product
    return ProductTermsResult(
        ok=True,
        scenario_id=scenario.id,
        company=scenario.company,
        product=p.name,
        rate_band=p.rate_band,
        amount_max_inr=p.amount_max_inr,
        tenure_months=p.tenure_months,
        fees=list(p.fees),
        figures=list(p.figures),
        callback_hours=scenario.compliance.callback_hours,
    )

"""I/O contracts for the deterministic grounding tools (brain/finance.py)."""

from __future__ import annotations

from pydantic import BaseModel, Field

from schemas.scenario import Fee, Figure, RateBand


class EmiRequest(BaseModel):
    principal_inr: float = Field(gt=0)
    annual_rate_pct: float = Field(gt=0)
    tenure_months: int = Field(gt=0)


class EmiResult(BaseModel):
    """calculate_emi output. ok=False -> error says why (e.g. rate outside the offered band)."""

    ok: bool
    error: str | None = None
    principal_inr: float | None = None
    annual_rate_pct: float | None = None
    tenure_months: int | None = None
    emi_inr: int | None = None  # rounded to the nearest rupee
    total_interest_inr: int | None = None
    total_payable_inr: int | None = None


class ProductTermsResult(BaseModel):
    """get_product_terms output: everything quotable for the active scenario."""

    ok: bool = True
    scenario_id: str
    company: str
    product: str
    rate_band: RateBand | None = None
    amount_max_inr: int | None = None
    tenure_months: tuple[int, int] | None = None
    fees: list[Fee] = Field(default_factory=list)
    figures: list[Figure] = Field(default_factory=list)
    callback_hours: tuple[int, int]

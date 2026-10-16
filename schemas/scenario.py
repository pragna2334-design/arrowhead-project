"""Scenario pack contract: one YAML per use case (scenarios/<id>/scenario.yaml).

A scenario owns everything use-case specific: company, product terms (the only figures the
agent may state), compliance rules, objection playbook and opening line. Global settings
(models, voice, prices) stay in config.yaml.
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field, model_validator


class RateBand(BaseModel):
    min_pct: float = Field(gt=0)
    max_pct: float = Field(gt=0)

    @model_validator(mode="after")
    def _ordered(self) -> RateBand:
        if self.min_pct > self.max_pct:
            raise ValueError("rate band min_pct > max_pct")
        return self


class Fee(BaseModel):
    """A quotable fee. kind=pct -> percent of amount; kind=inr -> flat rupees."""

    id: str  # e.g. "processing_fee"
    label: str  # how the agent says it, e.g. "processing fee"
    kind: Literal["pct", "inr"]
    value: float = Field(ge=0)


class Figure(BaseModel):
    """Any other quotable number (premium, sum insured, credit limit, reward rate...)."""

    id: str
    label: str
    value: float
    unit: Literal["inr", "pct", "months", "days", "years", "count"]


class ProductTerms(BaseModel):
    name: str  # "pre-approved personal loan"
    pitch: str  # one sentence of selling facts (no numbers; numbers live in the fields below)
    rate_band: RateBand | None = None  # interest/APR band, if the product has one
    amount_max_inr: int | None = None
    tenure_months: tuple[int, int] | None = None
    fees: list[Fee] = Field(default_factory=list)
    figures: list[Figure] = Field(default_factory=list)
    emi_enabled: bool = False  # expose calculate_emi (lending products only)


class ScenarioCompliance(BaseModel):
    callback_hours: tuple[int, int] = (10, 19)  # [start, end) local hours
    disclosure_terms: list[str]  # first agent turn must contain all of these (case-insensitive)
    forbidden_claims: list[str] = Field(default_factory=list)  # regexes, scenario-specific
    extra_rules: list[str] = Field(default_factory=list)  # plain-language rules for the prompt


class ObjectionPlay(BaseModel):
    id: str  # e.g. "rate_too_high", "premium_too_high"
    cues: list[str]  # customer phrases (Hinglish/Hindi/English) that signal it
    guidance: str  # how the agent should handle it (goes into prompt v3)


class ScenarioConfig(BaseModel):
    """scenarios/<id>/scenario.yaml. Personas live next to it in personas.yaml."""

    schema_version: int = 1
    id: str  # directory name, e.g. "personal_loan"
    title: str
    segment: Literal["lending", "insurance", "cards"]
    company: str  # fictional company the agent represents
    company_kind: str  # "an NBFC", "a general insurer", "a bank"
    customer_name: str  # default fictional customer
    product: ProductTerms
    compliance: ScenarioCompliance
    objections: list[ObjectionPlay] = Field(default_factory=list)
    # Opening template; placeholders: {agent_name} {verb} {company} {customer_name}
    opening: str

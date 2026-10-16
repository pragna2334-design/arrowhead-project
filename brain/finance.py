"""Deterministic grounding tools. PLACEHOLDER: implemented by the grounding work package.

Frozen interface (schemas/tools.py):
    calculate_emi(req: EmiRequest, scenario: ScenarioConfig) -> EmiResult
    get_product_terms(scenario: ScenarioConfig) -> ProductTermsResult
"""

from __future__ import annotations

from schemas import EmiRequest, EmiResult, ProductTermsResult, ScenarioConfig


def calculate_emi(req: EmiRequest, scenario: ScenarioConfig) -> EmiResult:
    raise NotImplementedError


def get_product_terms(scenario: ScenarioConfig) -> ProductTermsResult:
    raise NotImplementedError

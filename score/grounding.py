"""Grounded-figures rule. PLACEHOLDER: implemented by the grounding work package.

Frozen interface:
    check(t: Transcript, scenario: ScenarioConfig) -> tuple[bool | None, list[Violation]]
        True = every figure the agent stated is grounded; False = at least one hallucinated
        figure (Violation.rule == "grounded_figures", detail starts with "hallucinated figure");
        None = the agent stated no figures.
"""

from __future__ import annotations

from schemas import ScenarioConfig, Transcript, Violation


def check(t: Transcript, scenario: ScenarioConfig) -> tuple[bool | None, list[Violation]]:
    return None, []

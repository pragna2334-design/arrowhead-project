"""Scenario-specific rules. PLACEHOLDER: implemented by the scenario-packs work package.

Frozen interface:
    check(t: Transcript, scenario: ScenarioConfig) -> tuple[bool | None, list[Violation]]
        Applies scenario.compliance.forbidden_claims (and any scenario-specific checks) to agent
        turns. Violation.rule == "scenario_claims". None = scenario defines no such rules.
"""

from __future__ import annotations

from schemas import ScenarioConfig, Transcript, Violation


def check(t: Transcript, scenario: ScenarioConfig) -> tuple[bool | None, list[Violation]]:
    return None, []

"""Scenario packs: scenarios/<id>/scenario.yaml (+ personas.yaml). Selected by config
`scenario`, env SCENARIO, or load_config(scenario=...)."""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path

import yaml

from schemas import ScenarioConfig

SCENARIOS_DIR = Path(__file__).resolve().parent.parent / "scenarios"
DEFAULT_SCENARIO = "personal_loan"


def available_scenarios() -> list[str]:
    return sorted(p.parent.name for p in SCENARIOS_DIR.glob("*/scenario.yaml"))


def scenario_dir(scenario_id: str) -> Path:
    return SCENARIOS_DIR / scenario_id


def personas_path(scenario_id: str) -> Path:
    return scenario_dir(scenario_id) / "personas.yaml"


@lru_cache(maxsize=16)
def load_scenario(scenario_id: str = DEFAULT_SCENARIO) -> ScenarioConfig:
    path = scenario_dir(scenario_id) / "scenario.yaml"
    if not path.exists():
        raise FileNotFoundError(f"scenario {scenario_id!r} not found; have {available_scenarios()}")
    sc = ScenarioConfig.model_validate(yaml.safe_load(path.read_text()))
    if sc.id != scenario_id:
        raise ValueError(f"{path}: id {sc.id!r} does not match directory {scenario_id!r}")
    return sc

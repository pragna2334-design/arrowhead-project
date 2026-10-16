"""Config loading. One YAML file feeds the brain, the voice pipeline and the simulator."""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

from brain.scenario import DEFAULT_SCENARIO, load_scenario
from schemas import ScenarioConfig

DEFAULT_CONFIG_PATH = Path(__file__).resolve().parent.parent / "config.yaml"


@dataclass(frozen=True)
class BrainConfig:
    """Everything the brain needs. Identical for voice and text mode.

    Product/compliance fields are copied from the active scenario pack for convenience;
    `scenario` holds the full pack. rate_*/max_amount/tenure are None when the scenario's
    product has no such term (e.g. insurance).
    """

    lender_name: str  # scenario.company
    product: str
    rate_min_pct: float | None
    rate_max_pct: float | None
    max_amount_inr: int | None
    tenure_months: tuple[int, int] | None
    agent_name: str
    agent_gender: str  # "female" | "male"
    script: str  # "roman" | "devanagari"
    agent_model: str
    judge_model: str
    temperature: float
    max_output_tokens: int
    thinking: str  # "enabled" | "disabled"
    max_tool_rounds: int
    prompt_version: str
    callback_hours: tuple[int, int]  # [start, end) local hours, e.g. (10, 19)
    default_customer_name: str
    scenario: ScenarioConfig


@dataclass(frozen=True)
class AppConfig:
    """Full config: brain settings plus raw voice-side sections."""

    brain: BrainConfig
    stt: dict[str, Any] = field(default_factory=dict)
    tts: dict[str, Any] = field(default_factory=dict)
    voice: dict[str, Any] = field(default_factory=dict)
    sim: dict[str, Any] = field(default_factory=dict)
    score: dict[str, Any] = field(default_factory=dict)
    results_dir: str = "results"
    prices: dict[str, Any] = field(default_factory=dict)
    compare: dict[str, Any] = field(default_factory=dict)
    bench: dict[str, Any] = field(default_factory=dict)

    @property
    def scenario(self) -> ScenarioConfig:
        return self.brain.scenario


def load_config(
    path: str | Path | None = None,
    *,
    prompt_version: str | None = None,
    scenario: str | None = None,
) -> AppConfig:
    """Load config.yaml plus the active scenario pack.

    Overrides: `prompt_version` / env PROMPT_VERSION, `scenario` / env SCENARIO.
    """
    raw = yaml.safe_load(Path(path or os.getenv("CONFIG_PATH") or DEFAULT_CONFIG_PATH).read_text())
    override = prompt_version or os.getenv("PROMPT_VERSION")
    if override:
        raw["brain"]["prompt_version"] = override
    sc = load_scenario(scenario or os.getenv("SCENARIO") or raw.get("scenario") or DEFAULT_SCENARIO)
    b = raw["brain"]
    p = sc.product
    if b["agent_model"] == b["judge_model"]:
        raise ValueError("brain.judge_model must differ from brain.agent_model (self-grading bias)")
    brain = BrainConfig(
        lender_name=sc.company,
        product=p.name,
        rate_min_pct=p.rate_band.min_pct if p.rate_band else None,
        rate_max_pct=p.rate_band.max_pct if p.rate_band else None,
        max_amount_inr=p.amount_max_inr,
        tenure_months=p.tenure_months,
        agent_name=b["agent_name"],
        agent_gender=b["agent_gender"],
        script=b["script"],
        agent_model=b["agent_model"],
        judge_model=b["judge_model"],
        temperature=float(b["temperature"]),
        max_output_tokens=int(b["max_output_tokens"]),
        thinking=b["thinking"],
        max_tool_rounds=int(b["max_tool_rounds"]),
        prompt_version=str(b["prompt_version"]),
        callback_hours=sc.compliance.callback_hours,
        default_customer_name=sc.customer_name,
        scenario=sc,
    )
    return AppConfig(
        brain=brain,
        stt=raw.get("stt", {}),
        tts=raw.get("tts", {}),
        voice=raw.get("voice", {}),
        sim=raw.get("sim", {}),
        score=raw.get("score", {}),
        results_dir=raw.get("paths", {}).get("results_dir", "results"),
        prices=raw.get("prices", {}),
        compare=raw.get("compare", {}),
        bench=raw.get("bench", {}),
    )

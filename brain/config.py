"""Config loading. One YAML file feeds the brain, the voice pipeline and the simulator."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

DEFAULT_CONFIG_PATH = Path(__file__).resolve().parent.parent / "config.yaml"


@dataclass(frozen=True)
class BrainConfig:
    """Everything the brain needs. Identical for voice and text mode."""

    lender_name: str
    product: str
    rate_min_pct: float
    rate_max_pct: float
    max_amount_inr: int
    tenure_months: tuple[int, int]
    agent_name: str
    agent_gender: str  # "female" | "male"
    script: str  # "roman" | "devanagari"
    agent_model: str
    judge_model: str
    temperature: float
    max_output_tokens: int
    thinking: str  # "enabled" | "disabled"
    max_tool_rounds: int
    default_customer_name: str


@dataclass(frozen=True)
class AppConfig:
    """Full config: brain settings plus raw voice-side sections."""

    brain: BrainConfig
    stt: dict[str, Any] = field(default_factory=dict)
    tts: dict[str, Any] = field(default_factory=dict)
    voice: dict[str, Any] = field(default_factory=dict)


def load_config(path: str | Path | None = None) -> AppConfig:
    raw = yaml.safe_load(Path(path or DEFAULT_CONFIG_PATH).read_text())
    lender, b = raw["lender"], raw["brain"]
    if b["agent_model"] == b["judge_model"]:
        raise ValueError("brain.judge_model must differ from brain.agent_model (self-grading bias)")
    brain = BrainConfig(
        lender_name=lender["name"],
        product=lender["product"],
        rate_min_pct=float(lender["rate_min_pct"]),
        rate_max_pct=float(lender["rate_max_pct"]),
        max_amount_inr=int(lender["max_amount_inr"]),
        tenure_months=(int(lender["tenure_months"][0]), int(lender["tenure_months"][1])),
        agent_name=b["agent_name"],
        agent_gender=b["agent_gender"],
        script=b["script"],
        agent_model=b["agent_model"],
        judge_model=b["judge_model"],
        temperature=float(b["temperature"]),
        max_output_tokens=int(b["max_output_tokens"]),
        thinking=b["thinking"],
        max_tool_rounds=int(b["max_tool_rounds"]),
        default_customer_name=raw["customer_default"]["name"],
    )
    return AppConfig(
        brain=brain, stt=raw.get("stt", {}), tts=raw.get("tts", {}), voice=raw.get("voice", {})
    )

from __future__ import annotations

from pydantic import BaseModel, Field

from schemas.common import AttackType, Outcome
from schemas.score import CallScore


class PersonaRow(BaseModel):
    persona_id: str
    calls: int
    outcomes: dict[str, int]  # Outcome -> count
    violations: int
    judge_mean: float | None = None  # mean of the 4 judge dims over this persona's calls


class RedTeamRow(BaseModel):
    persona_id: str
    attack_type: AttackType
    call_id: str
    passed: bool
    reason: str


class Summary(BaseModel):
    """Written to results/<version>/summary.json."""

    schema_version: int = 1
    prompt_version: str
    generated_at: float
    agent_model: str
    judge_model: str | None = None
    n_calls: int
    outcomes: dict[Outcome, int]
    violation_counts: dict[str, int]  # RuleName -> number of violations
    calls_with_violations: int
    violation_rate: float  # calls_with_violations / n_calls
    consent_rate: float
    judge_means: dict[str, float] = Field(default_factory=dict)  # JudgeDim -> mean
    judge_distributions: dict[str, dict[str, int]] = Field(
        default_factory=dict
    )  # dim -> "1".."5" -> n
    per_persona: list[PersonaRow] = Field(default_factory=list)
    red_team: list[RedTeamRow] = Field(default_factory=list)
    calls: list[CallScore] = Field(default_factory=list)
    errored_calls: list[str] = Field(default_factory=list)  # excluded: stop_reason == error

from __future__ import annotations

from pydantic import BaseModel, Field

from schemas.common import AttackType, JudgeDim, Outcome, RuleName


class Violation(BaseModel):
    rule: RuleName
    turn_index: int | None = None  # index into Transcript.turns
    excerpt: str  # short, redacted transcript excerpt
    detail: str


class RuleResults(BaseModel):
    """True = passed, False = violated, None = not applicable for this call."""

    disclosure: bool
    dnc_honored: bool | None = None
    rate_in_range: bool
    no_guaranteed_approval: bool
    callback_in_hours: bool | None = None
    no_pii_leak: bool


class JudgeDimension(BaseModel):
    score: int = Field(ge=1, le=5)
    reason: str


class JudgeScores(BaseModel):
    objection_handling: JudgeDimension
    naturalness: JudgeDimension
    empathy: JudgeDimension
    hinglish_fluency: JudgeDimension

    def as_dict(self) -> dict[JudgeDim, JudgeDimension]:
        return {
            "objection_handling": self.objection_handling,
            "naturalness": self.naturalness,
            "empathy": self.empathy,
            "hinglish_fluency": self.hinglish_fluency,
        }


class CallScore(BaseModel):
    """Written to results/<version>/scores/<call_id>.json."""

    schema_version: int = 1
    call_id: str
    persona_id: str | None
    prompt_version: str
    turns: int  # total turns (agent + customer)
    outcome: Outcome
    disclosure_ok: bool
    consent_captured: bool
    objection_handled: bool | None = None  # None when no objection was raised
    rules: RuleResults
    compliance_violations: list[Violation] = Field(default_factory=list)
    judge: JudgeScores | None = None  # None if judge skipped/failed
    judge_model: str | None = None
    judge_error: str | None = None
    is_red_team: bool = False
    attack_type: AttackType | None = None
    red_team_passed: bool | None = None  # None for non-red-team calls

from __future__ import annotations

from pydantic import BaseModel, Field

from schemas.summary import Summary


class MetricDelta(BaseModel):
    a: float
    b: float
    delta: float  # b - a
    # Uncertainty (filled when per-call samples are available; None for single values).
    n_a: int | None = None
    n_b: int | None = None
    a_ci95: tuple[float, float] | None = None
    b_ci95: tuple[float, float] | None = None
    delta_ci95: tuple[float, float] | None = None
    significant: bool | None = None  # delta_ci95 excludes 0 (None if not computed)
    method: str | None = None  # e.g. "wilson", "newcombe", "bootstrap-2000"


class PersonaDiff(BaseModel):
    persona_id: str
    a_outcome: str | None
    b_outcome: str | None
    a_violations: int
    b_violations: int
    a_judge_mean: float | None = None
    b_judge_mean: float | None = None


class Comparison(BaseModel):
    """Written to results/compare_<a>_<b>.json by `make compare A=v1 B=v2`."""

    schema_version: int = 1
    a: str  # prompt version
    b: str
    generated_at: float
    violation_rate: MetricDelta
    consent_rate: MetricDelta
    outcomes: dict[str, MetricDelta]  # outcome -> share of calls (0..1)
    violation_counts: dict[str, MetricDelta]  # rule -> count
    judge_means: dict[str, MetricDelta]
    red_team_pass_rate: MetricDelta
    per_persona: list[PersonaDiff] = Field(default_factory=list)
    a_summary: Summary
    b_summary: Summary
    scenario: str = "personal_loan"
    repeats_per_persona: int | None = None
    cost_per_call: MetricDelta | None = None
    hallucinated_figure_rate: MetricDelta | None = None
    notes: list[str] = Field(
        default_factory=list
    )  # e.g. "differences marked n.s. are not significant"

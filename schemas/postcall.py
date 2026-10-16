from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field

from schemas.common import Mode, Mood

# Objection ids. personal_loan uses the original five; other scenarios define their own ids in
# scenario.yaml (objections[].id), so the field is a plain string. "other" = unclassified pushback.
Objection = str


class PostCallRecord(BaseModel):
    """CRM record produced after every call (deterministic, no LLM).

    Text mode: results/<version>/postcall/<call_id>.json. Voice: logs/calls/<call_id>.postcall.json.
    """

    schema_version: int = 1
    call_id: str
    mode: Mode
    prompt_version: str
    persona_id: str | None = None
    scenario: str = "personal_loan"
    created_at: float
    disposition: Literal[
        "interested_callback", "callback_requested", "declined", "dnc", "wrong_person", "incomplete"
    ]
    objections: list[Objection] = Field(default_factory=list)
    sentiment_trajectory: list[Mood] = Field(default_factory=list)  # one per customer turn
    final_sentiment: Mood | None = None
    consent_status: Literal["given", "not_given"]
    next_action: Literal["executive_callback", "retry_callback", "no_action", "suppress_number"]
    callback_time: str | None = None
    dnc: bool
    turns: int
    summary: str  # one-line, PII-redacted

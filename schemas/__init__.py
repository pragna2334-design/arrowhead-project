"""Frozen data contracts shared by brain, voice, sim, score, postcall, compare and report.

Every JSON file the project writes validates against one of these models. Bump
SCHEMA_VERSION on any breaking change.
"""

from schemas.compare import Comparison, MetricDelta, PersonaDiff
from schemas.latency import LatencyRecord
from schemas.layout import ResultsLayout
from schemas.postcall import PostCallRecord
from schemas.score import CallScore, JudgeDimension, JudgeScores, RuleResults, Violation
from schemas.summary import PersonaRow, RedTeamRow, Summary
from schemas.transcript import FinalState, Transcript, TranscriptToolCall, TranscriptTurn

SCHEMA_VERSION = 1

__all__ = [
    "SCHEMA_VERSION",
    "CallScore",
    "Comparison",
    "FinalState",
    "JudgeDimension",
    "JudgeScores",
    "LatencyRecord",
    "MetricDelta",
    "PersonaDiff",
    "PersonaRow",
    "PostCallRecord",
    "RedTeamRow",
    "ResultsLayout",
    "RuleResults",
    "Summary",
    "Transcript",
    "TranscriptToolCall",
    "TranscriptTurn",
    "Violation",
]

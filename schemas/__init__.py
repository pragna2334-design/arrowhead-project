"""Frozen data contracts shared by brain, voice, sim, score, postcall, compare and report.

Every JSON file the project writes validates against one of these models. Bump
SCHEMA_VERSION on any breaking change.
"""

from schemas.compare import Comparison, MetricDelta, PersonaDiff
from schemas.cost import CallUsage, CostRecord, Usage
from schemas.latency import LatencyRecord
from schemas.layout import ResultsLayout
from schemas.postcall import PostCallRecord
from schemas.scenario import (
    Fee,
    Figure,
    ObjectionPlay,
    ProductTerms,
    RateBand,
    ScenarioCompliance,
    ScenarioConfig,
)
from schemas.score import CallScore, JudgeDimension, JudgeScores, RuleResults, Violation
from schemas.summary import CostSummary, HallucinatedFigureRow, PersonaRow, RedTeamRow, Summary
from schemas.tools import EmiRequest, EmiResult, ProductTermsResult
from schemas.transcript import FinalState, Transcript, TranscriptToolCall, TranscriptTurn

SCHEMA_VERSION = 1

__all__ = [
    "SCHEMA_VERSION",
    "CallScore",
    "CallUsage",
    "Comparison",
    "CostRecord",
    "CostSummary",
    "EmiRequest",
    "EmiResult",
    "Fee",
    "Figure",
    "FinalState",
    "HallucinatedFigureRow",
    "JudgeDimension",
    "JudgeScores",
    "LatencyRecord",
    "MetricDelta",
    "ObjectionPlay",
    "PersonaDiff",
    "PersonaRow",
    "PostCallRecord",
    "ProductTerms",
    "ProductTermsResult",
    "RateBand",
    "RedTeamRow",
    "ResultsLayout",
    "RuleResults",
    "ScenarioCompliance",
    "ScenarioConfig",
    "Summary",
    "Transcript",
    "TranscriptToolCall",
    "TranscriptTurn",
    "Usage",
    "Violation",
]

import json
from pathlib import Path

import pytest

from schemas import (
    CallScore,
    Comparison,
    LatencyRecord,
    PostCallRecord,
    ResultsLayout,
    Summary,
    Transcript,
)

FX = Path(__file__).resolve().parent.parent / "schemas" / "fixtures"

CASES = [
    ("transcript.json", Transcript),
    ("transcript_dnc.json", Transcript),
    ("score.json", CallScore),
    ("summary.json", Summary),
    ("summary_v2.json", Summary),
    ("postcall.json", PostCallRecord),
    ("compare.json", Comparison),
]


@pytest.mark.parametrize("name,model", CASES)
def test_fixture_validates(name, model):
    obj = model.model_validate_json((FX / name).read_text())
    # round trip is lossless
    assert model.model_validate_json(obj.model_dump_json()) == obj


def test_latency_fixture_lines():
    lines = (FX / "latency.jsonl").read_text().splitlines()
    recs = [LatencyRecord.model_validate(json.loads(x)) for x in lines]
    assert {r.kind for r in recs} == {"greeting", "turn"}


def test_invalid_judge_score_rejected():
    raw = json.loads((FX / "score.json").read_text())
    raw["judge"]["empathy"]["score"] = 7
    with pytest.raises(ValueError):
        CallScore.model_validate(raw)


def test_layout_paths(tmp_path):
    lay = ResultsLayout(tmp_path, "v2")  # default scenario
    assert lay.transcripts == tmp_path / "personal_loan" / "v2" / "transcripts"
    assert lay.summary.name == "summary.json"
    assert lay.comparison("v3") == tmp_path / "personal_loan" / "compare_v2_v3.json"
    ins = ResultsLayout(tmp_path, "v3", scenario="health_insurance_renewal")
    assert ins.costs == tmp_path / "health_insurance_renewal" / "v3" / "costs"

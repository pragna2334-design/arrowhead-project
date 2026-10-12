"""Regression tests for the independent review findings."""

import asyncio
import json
import os
import pathlib
import time

from brain.config import load_config
from brain.postcall import write_postcall_records
from brain.redact import redact
from report.stats import load_latency
from schemas import CallScore, ResultsLayout, Summary, Transcript
from score.score import _needs_scoring
from score.score import run as score_run

FX = os.path.join(os.path.dirname(__file__), "..", "schemas", "fixtures")


def fx(name: str) -> str:
    return pathlib.Path(FX, name).read_text()


def put_transcript(lay: ResultsLayout, call_id: str, *, error: bool) -> None:
    t = Transcript.model_validate_json(fx("transcript.json"))
    t = t.model_copy(update={"call_id": call_id, "persona_id": "rt_guarantee_bait"})
    if error:
        t = t.model_copy(update={"stop_reason": "error", "error": "401", "turns": []})
    lay.transcripts.mkdir(parents=True, exist_ok=True)
    (lay.transcripts / f"{call_id}.json").write_text(t.model_dump_json())


def test_errored_calls_are_excluded_not_scored(tmp_path):
    lay = ResultsLayout(tmp_path, "v1")
    put_transcript(lay, "v1-ok-0", error=False)
    put_transcript(lay, "v1-bad-0", error=True)
    s: Summary = asyncio.run(score_run("v1", str(tmp_path), use_judge=False))
    assert s.n_calls == 1 and s.errored_calls == ["v1-bad-0"]
    assert not (lay.scores / "v1-bad-0.json").exists()
    assert all(r.call_id != "v1-bad-0" for r in s.red_team)  # no free red-team "pass"
    assert write_postcall_records("v1", str(tmp_path)) == 1


def test_rescored_and_postcall_refreshed_after_resimulation(tmp_path):
    lay = ResultsLayout(tmp_path, "v1")
    put_transcript(lay, "v1-x-0", error=False)
    asyncio.run(score_run("v1", str(tmp_path), use_judge=False))
    assert write_postcall_records("v1", str(tmp_path)) == 1
    assert write_postcall_records("v1", str(tmp_path)) == 0  # up to date
    future = time.time() + 10
    os.utime(lay.transcripts / "v1-x-0.json", (future, future))  # re-simulated
    assert write_postcall_records("v1", str(tmp_path)) == 1
    assert _needs_scoring(
        lay.scores / "v1-x-0.json", lay.transcripts / "v1-x-0.json", fresh=False, use_judge=False
    )


def test_rules_only_score_is_upgraded_when_judge_available(tmp_path):
    src = tmp_path / "t.json"
    src.write_text("{}")
    out = tmp_path / "s.json"
    score = CallScore.model_validate_json(fx("score.json"))
    out.write_text(score.model_copy(update={"judge": None, "judge_error": None}).model_dump_json())
    os.utime(src, (1, 1))
    assert not _needs_scoring(out, src, fresh=False, use_judge=False)
    assert _needs_scoring(out, src, fresh=False, use_judge=True)


def test_sim_exit_codes(monkeypatch, capsys):
    from sim import simulate

    monkeypatch.setenv("DEEPSEEK_API_KEY", "dummy")
    assert simulate.main(["--personas", "no_such_persona"]) == 2
    assert simulate.EXIT_CALLS_ERRORED == 3


def test_redaction_spoken_formats():
    assert redact("mera number 98 76 54 32 10 hai") == "mera number [PHONE] hai"
    assert redact("9 8 7 6 5 4 3 2 1 0") == "[PHONE]"
    assert redact("PAN: ABCDE 1234 F") == "PAN: [PAN]"
    assert redact("card 4111 1111 1111 1111") == "card [CARD]"
    for safe in ("Rs 5,00,000", "5 00 000", "12-10-2026 18:00", "11.5% se 18%", "12 to 60"):
        assert redact(safe) == safe


def test_latency_loader_skips_bad_lines_and_filters_version(tmp_path):
    rec = json.loads(fx("latency.jsonl").splitlines()[0])
    p = tmp_path / "l.jsonl"
    p.write_text(
        json.dumps({**rec, "prompt_version": "v1"})
        + "\n"
        + json.dumps({**rec, "prompt_version": "v2"})
        + "\n"
        + '{"truncated": \n'
    )
    assert len(load_latency(p)) == 2
    assert [r.prompt_version for r in load_latency(p, version="v2")] == ["v2"]


def test_make_client_has_no_hidden_retries(monkeypatch):
    from brain.agent import make_client

    monkeypatch.setenv("DEEPSEEK_API_KEY", "dummy")
    assert make_client().max_retries == 0
    assert load_config().brain.agent_model != load_config().brain.judge_model

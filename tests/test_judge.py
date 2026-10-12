"""Offline tests for score/judge.py with a stub OpenAI client (no network)."""

from __future__ import annotations

import asyncio
import json
from pathlib import Path
from types import SimpleNamespace

import httpx
import openai
import pytest

from schemas import Transcript
from score.judge import (
    Judge,
    JudgeParseError,
    build_messages,
    ensure_distinct_models,
    make_judge,
    parse_judge_output,
)

FX = Path(__file__).parent / "fixtures" / "score"

GOOD = {
    "objection_handling": {"score": 4, "reason": "Handled the rate objection honestly."},
    "naturalness": {"score": 5, "reason": "Short spoken turns."},
    "empathy": {"score": 3, "reason": "Polite."},
    "hinglish_fluency": {"score": 4, "reason": "Natural mixing."},
}


def load(name: str) -> Transcript:
    return Transcript.model_validate_json((FX / f"{name}.json").read_text())


def reply(content: str) -> SimpleNamespace:
    return SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content=content))])


class StubClient:
    """chat.completions.create returns/raises the queued items in order and records kwargs."""

    def __init__(self, *items):
        self.items = list(items)
        self.calls: list[dict] = []
        self.chat = SimpleNamespace(completions=SimpleNamespace(create=self._create))

    async def _create(self, **kwargs):
        self.calls.append(kwargs)
        item = self.items.pop(0)
        if isinstance(item, Exception):
            raise item
        return reply(item)


def api_error(cls, status: int):
    resp = httpx.Response(status, request=httpx.Request("POST", "https://api.example/v1"))
    return cls(f"status {status}", response=resp, body=None)


async def _nosleep(_: float) -> None:
    return None


def make(client, **kw) -> Judge:
    kw.setdefault("rpm", 0)
    return Judge(client, "deepseek-pro", agent_model="deepseek-flash", sleep=_nosleep, **kw)


def run(coro):
    return asyncio.run(coro)


# ---------------------------------------------------------------- parsing


def test_parse_plain_json():
    r = parse_judge_output(json.dumps(GOOD))
    assert r.scores.naturalness.score == 5
    assert r.scores.objection_handling.reason.startswith("Handled")
    assert r.held_rules is None


def test_parse_fenced_and_prose_and_red_team():
    body = {**GOOD, "held_rules": True, "held_rules_reason": "Refused."}
    r = parse_judge_output("Here you go:\n```json\n" + json.dumps(body) + "\n```\nThanks")
    assert r.held_rules is True and r.held_rules_reason == "Refused."
    r = parse_judge_output("Result: " + json.dumps({"scores": GOOD, "held_rules": "false"}))
    assert r.held_rules is False
    flat = {"objection_handling": 2, "naturalness": "3", "empathy": 4.0, "hinglish_fluency": 5}
    r = parse_judge_output(json.dumps(flat))
    assert [d.score for d in r.scores.as_dict().values()] == [2, 3, 4, 5]


@pytest.mark.parametrize(
    "text",
    [
        "not json",
        json.dumps({**GOOD, "empathy": {"score": 7, "reason": "x"}}),
        json.dumps({k: v for k, v in GOOD.items() if k != "naturalness"}),
    ],
)
def test_parse_errors(text):
    with pytest.raises(JudgeParseError):
        parse_judge_output(text)


# ---------------------------------------------------------------- model separation


def test_models_must_differ():
    with pytest.raises(ValueError, match="must differ"):
        ensure_distinct_models("deepseek-flash", "deepseek-flash")
    with pytest.raises(ValueError):
        Judge(StubClient(), "same", agent_model="same")
    cfg = SimpleNamespace(
        brain=SimpleNamespace(
            agent_model="m",
            judge_model="m",
            lender_name="Vriddhi Finance",
            rate_min_pct=11.5,
            rate_max_pct=18.0,
        ),
        score={"judge_temperature": 0.0, "rpm": 8},
    )
    with pytest.raises(ValueError):
        make_judge(cfg, client=StubClient())
    cfg.brain.judge_model = "other"
    assert make_judge(cfg, client=StubClient()).model == "other"


def test_transcript_from_judge_model_is_not_judged():
    t = load("compliant").model_copy(update={"agent_model": "deepseek-pro"})
    client = StubClient(json.dumps(GOOD))
    r = run(make(client).score(t))
    assert r.scores is None and "must differ" in r.error
    assert client.calls == []


# ---------------------------------------------------------------- requests


def test_request_shape_and_no_agent_prompt():
    client = StubClient(json.dumps(GOOD))
    r = run(
        make(client, temperature=0.0).score(load("compliant"), persona_description="Rate shopper")
    )
    assert r.error is None and r.scores.empathy.score == 3
    kw = client.calls[0]
    assert kw["model"] == "deepseek-pro"
    assert kw["temperature"] == 0.0
    assert kw["extra_body"] == {"thinking": {"type": "disabled"}}
    assert kw["response_format"] == {"type": "json_object"}
    text = json.dumps(kw["messages"], ensure_ascii=False)
    assert "Rate shopper" in text and "Rahul" in text
    assert "Call flow" not in text and "# Hard compliance rules" not in text  # no agent prompt
    assert "held_rules" not in text


def test_red_team_prompt_asks_for_held_rules():
    msgs = build_messages(
        load("rt_prompt_injection_pass"),
        lender="Vriddhi Finance",
        rate_min=11.5,
        rate_max=18,
        attack_type="prompt_injection",
    )
    assert "held_rules" in msgs[1]["content"] and "prompt_injection" in msgs[1]["content"]


def test_retry_on_429_and_5xx():
    slept: list[float] = []

    async def sleep(s: float) -> None:
        slept.append(s)

    client = StubClient(
        api_error(openai.RateLimitError, 429),
        api_error(openai.InternalServerError, 503),
        json.dumps(GOOD),
    )
    j = Judge(client, "deepseek-pro", agent_model="deepseek-flash", rpm=0, sleep=sleep)
    r = run(j.score(load("compliant")))
    assert r.scores is not None
    assert slept == [2, 4]


def test_gives_up_after_retries():
    errs = [api_error(openai.RateLimitError, 429) for _ in range(6)]
    r = run(make(StubClient(*errs)).score(load("compliant")))
    assert r.scores is None and "RateLimitError" in r.error


def test_non_retryable_error_sets_judge_error():
    r = run(make(StubClient(api_error(openai.AuthenticationError, 401))).score(load("compliant")))
    assert r.scores is None and r.error.startswith("AuthenticationError")


def test_json_mode_rejected_falls_back_to_text():
    client = StubClient(
        api_error(openai.BadRequestError, 400), "```json\n" + json.dumps(GOOD) + "```"
    )
    r = run(make(client).score(load("compliant")))
    assert r.scores is not None
    assert "response_format" in client.calls[0] and "response_format" not in client.calls[1]


def test_malformed_output_reasked_once_then_error():
    r = run(make(StubClient("garbage", json.dumps(GOOD))).score(load("compliant")))
    assert r.scores is not None
    r = run(make(StubClient("garbage", "still garbage")).score(load("compliant")))
    assert r.scores is None and "JudgeParseError" in r.error

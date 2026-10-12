"""Offline tests for the text-mode simulator. Stub LLM clients only; never touches the network."""

from __future__ import annotations

import asyncio
import json
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace
from typing import get_args

import httpx
import pytest
from openai import APIStatusError

from brain.config import load_config
from schemas import ResultsLayout, Transcript
from schemas.common import AttackType
from sim import customer as customer_mod
from sim.customer import HANGUP, CustomerSim, flip_roles
from sim.ratelimit import RateLimiter
from sim.simulate import (
    PERSONA_KEYS,
    load_personas,
    make_call_id,
    run_sim,
    select_personas,
)

PERSONA = {
    "id": "unit_persona",
    "customer_name": "Rahul Verma",
    "language": "hinglish",
    "description": "Test customer.",
    "behaviour": "Agree to everything.",
    "red_team": False,
    "attack_type": None,
}


# ---------------------------------------------------------------- stubs


def _msg(content=None, tool=None, args=None):
    calls = None
    if tool:
        calls = [
            SimpleNamespace(
                id=f"call_{tool}",
                function=SimpleNamespace(name=tool, arguments=json.dumps(args or {})),
            )
        ]
    return SimpleNamespace(
        choices=[SimpleNamespace(message=SimpleNamespace(content=content, tool_calls=calls))]
    )


class Script:
    """Async `create(**kw)`: agent requests (they carry `tools`) and customer requests pop from
    separate queues. A queue item may be an Exception to raise. Empty queue = default reply."""

    def __init__(self, agent=(), customer=(), agent_default=None, customer_default=None):
        self.agent = list(agent)
        self.customer = list(customer)
        self.agent_default = agent_default or _msg("Ji, aur bataiye?")
        self.customer_default = customer_default or _msg("Hmm, theek hai.")
        self.requests: list[dict] = []

    async def create(self, **kw):
        self.requests.append(kw)
        is_agent = "tools" in kw
        queue = self.agent if is_agent else self.customer
        item = (
            queue.pop(0) if queue else (self.agent_default if is_agent else self.customer_default)
        )
        if isinstance(item, Exception):
            raise item
        return item


def client(script: Script):
    return SimpleNamespace(chat=SimpleNamespace(completions=script))


async def _noop():
    return None


def _app(max_turns=16):
    app = load_config(prompt_version="v1")
    return replace(app, sim={**app.sim, "max_turns": max_turns})


def _run(tmp_path, script, personas=None, *, max_turns=16, fresh=False):
    c = client(script)
    return asyncio.run(
        run_sim(
            _app(max_turns),
            personas=personas or [PERSONA],
            fresh=fresh,
            results_dir=tmp_path,
            agent_client=c,
            customer_client=c,
            limiter=_noop,
            log=lambda *_: None,
        )
    )


def _transcript(tmp_path, call_id="v1-unit_persona-1") -> Transcript:
    path = ResultsLayout(Path(tmp_path), "v1").transcripts / f"{call_id}.json"
    return Transcript.model_validate_json(path.read_text())


OPENING = _msg(
    "Namaste! Main Priya, Vriddhi Finance ki AI assistant. Kya Rahul Verma ji baat kar rahe hain?"
)


# ---------------------------------------------------------------- personas


def test_personas_file_valid():
    personas = load_personas()
    assert len(personas) == 25
    ids = [p["id"] for p in personas]
    assert len(set(ids)) == 25
    for p in personas:
        assert set(PERSONA_KEYS) <= set(p), p["id"]
        assert p["id"].replace("_", "").isalnum() and p["id"] == p["id"].lower()
        assert p["language"] in ("hinglish", "hindi", "english")
        assert p["description"].strip() and p["behaviour"].strip()
        assert isinstance(p["red_team"], bool)
        assert p["red_team"] == p["id"].startswith("rt_")
        assert (p["attack_type"] is None) == (not p["red_team"])
    red = [p for p in personas if p["red_team"]]
    assert len(red) == 5
    assert sorted(p["attack_type"] for p in red) == sorted(get_args(AttackType))


def test_select_personas():
    personas = load_personas()
    assert len(select_personas(personas, 3)) == 3
    assert [p["id"] for p in select_personas(personas, None, ["rt_impersonation"])] == [
        "rt_impersonation"
    ]
    with pytest.raises(ValueError):
        select_personas(personas, None, ["nope"])
    assert make_call_id("v1", "a/b c", 2) == "v1-a_b_c-2"


# ---------------------------------------------------------------- customer LLM


def test_customer_prompt_and_role_flip():
    script = Script(customer=[_msg("Haan bolo.")])
    sim = CustomerSim(PERSONA, model="cust-model", temperature=0.8, client=client(script))
    turns = [
        SimpleNamespace(role="agent", text="Hello"),
        SimpleNamespace(role="customer", text="Hi"),
    ]
    turns.append(SimpleNamespace(role="agent", text="Loan chahiye?"))
    assert asyncio.run(sim.reply(turns)) == "Haan bolo."
    req = script.requests[0]
    assert req["model"] == "cust-model" and req["temperature"] == 0.8
    assert req["extra_body"] == {"thinking": {"type": "disabled"}}
    assert [m["role"] for m in req["messages"]] == ["system", "user", "assistant", "user"]
    assert HANGUP in req["messages"][0]["content"]
    assert flip_roles([SimpleNamespace(role="customer", text="x")]) == [
        {"role": "assistant", "content": "x"}
    ]


def test_customer_retries_429(monkeypatch):
    slept: list[float] = []

    async def fake_sleep(s):
        slept.append(s)

    monkeypatch.setattr(customer_mod.asyncio, "sleep", fake_sleep)
    req = httpx.Request("POST", "https://example.invalid")
    err = APIStatusError("rate", response=httpx.Response(429, request=req), body=None)
    calls = []

    async def limiter():
        calls.append(1)

    script = Script(customer=[err, err, _msg(HANGUP)])
    sim = CustomerSim(
        PERSONA, model="m", temperature=0.5, client=client(script), before_request=limiter
    )
    assert asyncio.run(sim.reply([])) == HANGUP
    assert slept == [2, 4] and len(calls) == 3


# ---------------------------------------------------------------- full calls


def test_full_call_agent_ends(tmp_path):
    script = Script(
        agent=[
            OPENING,
            _msg("Aapke liye 11.5% se loan offer hai. Executive call kar sakta hai?"),
            _msg(None, "record_consent"),
            _msg("Kab call karein?"),
            _msg(None, "book_callback", {"day": "tomorrow", "time": "11:00"}),
            _msg(None, "end_call", {"reason": "callback_booked"}),
        ],
        customer=[_msg("Haan, main Rahul."), _msg("Haan bilkul, chahiye."), _msg("Kal 11 baje.")],
    )
    out = _run(tmp_path, script)
    assert out == {"v1-unit_persona-1": "agent_ended"}
    tr = _transcript(tmp_path)
    assert tr.mode == "text" and tr.persona_id == "unit_persona" and tr.prompt_version == "v1"
    assert tr.final_state.consent is True
    assert tr.final_state.callback_time == "tomorrow 11:00"
    assert tr.final_state.end_reason == "callback_booked"
    assert [t.role for t in tr.turns] == [
        "agent",
        "customer",
        "agent",
        "customer",
        "agent",
        "customer",
        "agent",
    ]
    assert [c.name for c in tr.tool_calls] == ["record_consent", "book_callback", "end_call"]
    state = json.loads(ResultsLayout(tmp_path, "v1").sim_state.read_text())
    assert state["finished"] == ["v1-unit_persona-1"]
    assert state["calls"]["v1-unit_persona-1"]["stop_reason"] == "agent_ended"
    # Customer requests never carry tools; agent ones always do.
    assert sum("tools" not in r for r in script.requests) == 3


def test_customer_hangup(tmp_path):
    script = Script(agent=[OPENING], customer=[_msg("Kaun?"), _msg(f"  {HANGUP} ")])
    assert _run(tmp_path, script) == {"v1-unit_persona-1": "customer_hung_up"}
    tr = _transcript(tmp_path)
    assert [t.text for t in tr.turns if t.role == "customer"] == ["Kaun?"]
    assert tr.turns[-1].role == "agent"
    assert all(HANGUP not in t.text for t in tr.turns)


def test_max_turns(tmp_path):
    script = Script(agent=[OPENING])
    assert _run(tmp_path, script, max_turns=3) == {"v1-unit_persona-1": "max_turns"}
    tr = _transcript(tmp_path)
    assert sum(t.role == "customer" for t in tr.turns) == 3
    assert len(tr.turns) == 7


def test_error_still_writes_transcript(tmp_path):
    script = Script(agent=[OPENING, RuntimeError("boom")], customer=[_msg("Haan?")])
    assert _run(tmp_path, script) == {"v1-unit_persona-1": "error"}
    tr = _transcript(tmp_path)
    assert tr.stop_reason == "error" and "boom" in tr.error
    assert len(tr.turns) == 2  # opening + the customer line that was being answered
    state = json.loads(ResultsLayout(tmp_path, "v1").sim_state.read_text())
    assert state["errored"] == ["v1-unit_persona-1"]


def test_resume_skips_done_and_retries_errors(tmp_path):
    p2 = {**PERSONA, "id": "second"}
    script = Script(
        agent=[OPENING, _msg(None, "end_call", {"reason": "declined"}), RuntimeError("down")],
        customer=[_msg("Nahi chahiye.")],
    )
    first = _run(tmp_path, script, [PERSONA, p2])
    assert first == {"v1-unit_persona-1": "agent_ended", "v1-second-1": "error"}

    script2 = Script(agent=[_msg(None, "end_call", {"reason": "wrong_person"})])
    second = _run(tmp_path, script2, [PERSONA, p2])
    assert second == {"v1-unit_persona-1": "skipped", "v1-second-1": "agent_ended"}
    assert len(script2.requests) == 1  # only the errored call was re-run

    script3 = Script(agent=[_msg(None, "end_call", {"reason": "other"})] * 2)
    assert set(_run(tmp_path, script3, [PERSONA, p2], fresh=True).values()) == {"agent_ended"}
    state = json.loads(ResultsLayout(tmp_path, "v1").sim_state.read_text())
    assert state["finished"] == ["v1-second-1", "v1-unit_persona-1"] and state["errored"] == []


def test_transcripts_validate_and_no_scoring_outputs(tmp_path):
    script = Script(agent=[OPENING], customer=[_msg(HANGUP)])
    _run(tmp_path, script)
    layout = ResultsLayout(tmp_path, "v1")
    for f in layout.transcripts.glob("*.json"):
        Transcript.model_validate_json(f.read_text())
    assert not layout.scores.exists() and not layout.postcall.exists()


# ---------------------------------------------------------------- rate limiter


def test_rate_limiter_spacing_with_fake_clock():
    now = [100.0]
    slept: list[float] = []

    async def sleep(s):
        slept.append(s)
        now[0] += s

    lim = RateLimiter(6, clock=lambda: now[0], sleep=sleep)  # 10s apart

    async def go():
        stamps = []
        for _ in range(4):
            await lim()
            stamps.append(now[0])
        now[0] += 25  # idle longer than the interval: next request goes straight through
        await lim()
        stamps.append(now[0])
        return stamps

    stamps = asyncio.run(go())
    assert stamps == [100.0, 110.0, 120.0, 130.0, 155.0]
    assert slept == [10.0, 10.0, 10.0]


def test_rate_limiter_disabled():
    lim = RateLimiter(0, clock=lambda: 0.0, sleep=lambda s: pytest.fail("slept"))
    asyncio.run(lim.wait())
    asyncio.run(lim.wait())

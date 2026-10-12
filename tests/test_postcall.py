"""Offline tests for brain.postcall."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from brain.postcall import (
    build_postcall_record,
    detect_objections,
    main,
    write_postcall_records,
    write_voice_records,
)
from schemas import PostCallRecord, Transcript

ROOT = Path(__file__).resolve().parent.parent
SCHEMA_FIX = ROOT / "schemas" / "fixtures"
FIX = Path(__file__).resolve().parent / "fixtures" / "postcall"


def _load(path: Path) -> Transcript:
    return Transcript.model_validate_json(path.read_text(encoding="utf-8"))


def _make(
    customer: list[tuple[str, str | None]],
    *,
    consent: bool | None = None,
    callback_time: str | None = None,
    dnc: bool = False,
    ended: bool = True,
    end_reason: str | None = None,
    call_id: str = "t0001",
) -> Transcript:
    turns = []
    ts = 1.0
    for text, mood in customer:
        turns.append({"role": "agent", "text": "Namaste ji.", "ts": ts})
        turns.append({"role": "customer", "text": text, "ts": ts + 1, "sentiment": mood})
        ts += 2
    return Transcript.model_validate(
        {
            "call_id": call_id,
            "mode": "text",
            "prompt_version": "v1",
            "agent_model": "m",
            "persona_id": "p",
            "customer_name": "Rahul Verma",
            "started_at": 0.0,
            "ended_at": ts,
            "stop_reason": "agent_ended",
            "turns": turns,
            "final_state": {
                "consent": consent,
                "callback_time": callback_time,
                "dnc": dnc,
                "ended": ended,
                "end_reason": end_reason,
            },
        }
    )


def test_matches_schema_fixture() -> None:
    t = _load(SCHEMA_FIX / "transcript.json")
    expected = PostCallRecord.model_validate_json((SCHEMA_FIX / "postcall.json").read_text())
    got = build_postcall_record(t, created_at=expected.created_at)
    assert got == expected


def test_dnc_fixture() -> None:
    rec = build_postcall_record(_load(SCHEMA_FIX / "transcript_dnc.json"), created_at=1.0)
    assert rec.disposition == "dnc"
    assert rec.next_action == "suppress_number"
    assert rec.dnc is True
    assert rec.objections == ["number_source"]
    assert rec.consent_status == "not_given"
    assert rec.sentiment_trajectory == ["annoyed"]


@pytest.mark.parametrize(
    ("state", "disposition", "next_action", "consent_status"),
    [
        (
            {"consent": True, "callback_time": "tomorrow 18:00", "end_reason": "callback_booked"},
            "interested_callback",
            "executive_callback",
            "given",
        ),
        (
            {"callback_time": "Saturday 11:00", "end_reason": "callback_booked"},
            "callback_requested",
            "retry_callback",
            "not_given",
        ),
        (
            {"consent": False, "callback_time": "Saturday 11:00"},
            "callback_requested",
            "retry_callback",
            "not_given",
        ),
        ({"end_reason": "declined"}, "declined", "no_action", "not_given"),
        ({"end_reason": "completed", "consent": True}, "declined", "no_action", "given"),
        ({"end_reason": "wrong_person"}, "wrong_person", "no_action", "not_given"),
        (
            {"dnc": True, "end_reason": "dnc", "callback_time": "tomorrow 12:00"},
            "dnc",
            "suppress_number",
            "not_given",
        ),
        ({"ended": False}, "incomplete", "no_action", "not_given"),
    ],
)
def test_disposition_matrix(
    state: dict, disposition: str, next_action: str, consent_status: str
) -> None:
    rec = build_postcall_record(_make([("Haan ji", "calm")], **state), created_at=1.0)
    assert rec.disposition == disposition
    assert rec.next_action == next_action
    assert rec.consent_status == consent_status
    assert rec.callback_time == state.get("callback_time")
    assert rec.dnc is state.get("dnc", False)


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        # Hinglish (Roman)
        ("Rate bahut zyada hai yaar", ["rate_too_high"]),
        ("Abhi busy hoon, baad mein call karo", ["call_later"]),
        ("Mujhe nahi chahiye, interested nahi hoon", ["not_interested"]),
        ("Pehle se ek loan chal raha hai", ["already_has_loan"]),
        ("Aapko mera number kahan se mila?", ["number_source"]),
        # Hindi (Devanagari)
        ("ब्याज बहुत ज़्यादा है", ["rate_too_high"]),
        ("अभी व्यस्त हूँ, बाद में कीजिए", ["call_later"]),
        ("मुझे नहीं चाहिए", ["not_interested"]),
        ("मेरा पहले से एक लोन चल रहा है", ["already_has_loan"]),
        ("आपको मेरा नंबर कहाँ से मिला?", ["number_source"]),
        # English
        ("That interest rate is too high for me", ["rate_too_high"]),
        ("I'm busy, call me later", ["call_later"]),
        ("Sorry, not interested", ["not_interested"]),
        ("I already have a home loan", ["already_has_loan"]),
        ("Where did you get my number?", ["number_source"]),
        # Unknown but clearly negative pushback -> other
        ("Yeh sab fraud hai", ["other"]),
        ("यह बकवास है", ["other"]),
        # Neutral / positive -> nothing
        ("Haan, main Rahul bol raha hoon.", []),
        ("Theek hai, haan check karwa do, chahiye mujhe.", []),
        ("Kal shaam 6 baje.", []),
        ("What is the insurance premium?", []),
    ],
)
def test_detect_objections(text: str, expected: list[str]) -> None:
    assert detect_objections([text]) == expected


def test_objections_dedup_first_seen_order() -> None:
    texts = [
        "Mera number kahan se mila?",
        "Rate bahut zyada hai, aur pehle se loan chal raha hai",
        "Rate zyada hai bola na",
        "nonsense",  # 'other' only when nothing specific matched in that turn
    ]
    assert detect_objections(texts) == [
        "number_source",
        "rate_too_high",
        "already_has_loan",
        "other",
    ]


def test_objections_ignore_agent_turns() -> None:
    t = _make([("Haan ji", "calm")], end_reason="declined")
    t.turns[0].text = "Rate bahut zyada lag raha hai? Number kahan se mila, yeh bataun?"
    assert build_postcall_record(t, created_at=1.0).objections == []


def test_sentiment_trajectory_skips_none() -> None:
    t = _make(
        [("Haan", "calm"), ("hmm", None), ("Kitna milega?", "interested"), ("ok", "annoyed")],
        end_reason="declined",
    )
    rec = build_postcall_record(t, created_at=1.0)
    assert rec.sentiment_trajectory == ["calm", "interested", "annoyed"]
    assert rec.final_sentiment == "annoyed"
    assert rec.turns == 8


def test_no_customer_turns() -> None:
    rec = build_postcall_record(_make([], ended=False), created_at=1.0)
    assert rec.sentiment_trajectory == []
    assert rec.final_sentiment is None
    assert rec.objections == []
    assert rec.summary == "Call ended before any outcome."


def test_summary_is_redacted_and_one_line() -> None:
    t = _make(
        [("Busy hoon, baad mein", "annoyed")],
        callback_time="call 9876543210 tomorrow 18:00",
        end_reason="callback_booked",
    )
    rec = build_postcall_record(t, created_at=1.0)
    assert "9876543210" not in rec.summary
    assert "[PHONE]" in rec.summary
    assert "\n" not in rec.summary
    assert rec.summary.startswith("Customer asked to be called later, asked for a callback")


def test_created_at_defaults_to_now() -> None:
    rec = build_postcall_record(_make([("Haan", "calm")]))
    assert rec.created_at > 1_700_000_000


def test_extra_fixtures_validate() -> None:
    for path in sorted(FIX.glob("*.transcript.json")):
        expected_path = path.with_name(path.name.replace(".transcript.json", ".expected.json"))
        expected = json.loads(expected_path.read_text(encoding="utf-8"))
        rec = build_postcall_record(_load(path), created_at=1.0)
        PostCallRecord.model_validate(rec.model_dump(mode="json"))
        for key, value in expected.items():
            assert getattr(rec, key) == value, (path.name, key)


def _seed_results(tmp_path: Path) -> Path:
    tdir = tmp_path / "results" / "v9" / "transcripts"
    tdir.mkdir(parents=True)
    for name in ("transcript.json", "transcript_dnc.json"):
        (tdir / name).write_text((SCHEMA_FIX / name).read_text(encoding="utf-8"))
    return tmp_path / "results"


def test_write_postcall_records(tmp_path: Path) -> None:
    results = _seed_results(tmp_path)
    assert write_postcall_records("v9", str(results)) == 2
    out = results / "v9" / "postcall"
    files = sorted(p.name for p in out.glob("*.json"))
    assert files == ["fx000001.json", "fx000002.json"]
    for p in out.glob("*.json"):
        PostCallRecord.model_validate_json(p.read_text(encoding="utf-8"))
    assert write_postcall_records("v9", str(results)) == 0  # skip existing
    assert write_postcall_records("v9", str(results), fresh=True) == 2
    assert write_postcall_records("nope", str(results)) == 0


def test_cli(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    results = _seed_results(tmp_path)
    assert main(["--version", "v9", "--results", str(results)]) == 0
    assert "wrote 2 record(s)" in capsys.readouterr().out
    assert main(["--version", "v9", "--results", str(results)]) == 0
    assert "wrote 0 record(s)" in capsys.readouterr().out
    assert main(["--version", "v9", "--results", str(results), "--fresh"]) == 0
    assert "wrote 2 record(s)" in capsys.readouterr().out


def test_write_voice_records(tmp_path: Path) -> None:
    t = _make(
        [("Mera number 98765 43210 hai, baad mein call karo", "annoyed")],
        callback_time="tomorrow 12:00",
        end_reason="callback_booked",
        call_id="voice42",
    ).model_copy(update={"mode": "voice", "persona_id": None})
    t_path, p_path = write_voice_records(t, tmp_path / "logs" / "calls")
    assert t_path.name == "voice42.transcript.json"
    assert p_path.name == "voice42.postcall.json"
    saved = Transcript.model_validate_json(t_path.read_text(encoding="utf-8"))
    assert "98765" not in t_path.read_text(encoding="utf-8")
    assert saved.turns[1].text == "Mera number [PHONE] hai, baad mein call karo"
    rec = PostCallRecord.model_validate_json(p_path.read_text(encoding="utf-8"))
    assert rec.mode == "voice"
    assert rec.disposition == "callback_requested"
    assert rec.objections == ["call_later"]

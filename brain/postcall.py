"""Post-call intelligence: turn a finished Transcript into a CRM-ready PostCallRecord.

Deterministic and offline (keyword rules, no LLM), so it costs nothing and can be re-run.

    python -m brain.postcall [--version v1] [--results results] [--fresh]
"""

from __future__ import annotations

import argparse
import json
import re
import time
from collections.abc import Iterable
from pathlib import Path
from typing import get_args

from brain.redact import redact, redact_obj
from schemas import PostCallRecord, ResultsLayout, Transcript
from schemas.postcall import Objection

# --- objection detection -------------------------------------------------------------------

# Each cue is a regex fragment matched against lower-cased customer text. Latin cues are
# wrapped in word boundaries (so "later" does not hit "translater"); Devanagari cues are not.
_OBJECTION_CUES: dict[Objection, tuple[str, ...]] = {
    "rate_too_high": (
        (
            r"(?:rate|rates|interest|byaj|byaaj|vyaj)\b.{0,30}\b(?:zyada|jyada|zyaada|jyaada|"
            r"bahut|high|higher|zada|kam karo|kam kijiye|kam hai kya|too much)"
        ),
        r"(?:too|very|bahut|kaafi|kafi) (?:high|zyada|jyada|mehnga|mehenga|costly|expensive)",
        r"mehnga|mehenga|mahenga|expensive|costly",
        r"(?:better|lower|kam|sasta) (?:rate|interest|byaj)",
        r"\d+(?:\.\d+)?\s*%.{0,25}\b(?:bola|diya|de raha|offer|mil raha)",
        "ब्याज.{0,20}(?:ज़्यादा|ज्यादा|बहुत)",
        "रेट.{0,20}(?:ज़्यादा|ज्यादा|बहुत)",
        "महंगा|महँगा",
    ),
    "call_later": (
        r"baad (?:mein|me|main)",
        r"abhi (?:busy|time nahi|time nahin|baat nahi|nahi kar sakta|nahi kar sakti)",
        r"busy (?:hoon|hu|hun|hai)|i am busy|i'm busy",
        r"call (?:me )?(?:back )?later|later please|some other time|not now|not a good time",
        r"meeting (?:mein|me) (?:hoon|hu|hun)|driving",
        r"(?:kal|parson|shaam ko|evening) (?:call|phone) (?:karo|kijiye|karna)",
        "बाद में|अभी व्यस्त|अभी समय नहीं|अभी टाइम नहीं|बिज़ी|बिजी",
    ),
    "not_interested": (
        r"not interested|interested nahi|interested nahin|interest nahi|no interest",
        r"nahi chahiye|nahin chahiye|nai chahiye|chahiye nahi|nahi lena|nahin lena|lena nahi",
        r"(?:zaroorat|zarurat|jarurat|jaroorat) nahi|no need|don'?t need|do not need",
        r"don'?t want|do not want|not looking",
        "रुचि नहीं|इंटरेस्ट नहीं|नहीं चाहिए|ज़रूरत नहीं|जरूरत नहीं|नहीं लेना",
    ),
    "already_has_loan": (
        r"already (?:have|has|taken|got|running)\b.{0,20}\bloan",
        r"(?:pehle se|pahle se|already)\b.{0,20}\bloan",
        r"loan (?:chal raha|chal rahi|already|pehle se|pahle se|le rakha|liya hua)",
        r"existing loan|current loan|ek loan hai|another loan",
        "पहले से.{0,15}लोन|लोन चल रहा|लोन ले रखा|पहले से.{0,15}ऋण",
    ),
    "number_source": (
        r"number (?:kahan|kaha|kidhar|kaise|kis se|kisne)",
        r"(?:kahan|kaha|kaise) se (?:mila|liya)\b.{0,15}number",
        r"(?:where|how) did you get (?:my|this) (?:number|contact)",
        r"got my number|my number from",
        "नंबर कहाँ|नंबर कहां|नंबर कैसे|नंबर किसने",
    ),
}

# Clearly negative pushback that matches none of the specific objections -> "other".
# Stop-calling requests land here too; the DNC itself is captured by disposition/dnc.
_NEGATIVE_CUES: tuple[str, ...] = (
    r"fraud|scam|spam|cheat|dhokha|bharosa nahi|trust nahi|don'?t trust",
    r"bakwas|faltu|nonsense|rubbish|waste|pareshan|irritat|annoying|disturb",
    r"stop calling|don'?t call|do not call|call mat|phone mat|mat karo|band karo|bas karo",
    r"complaint|shikayat|remove my number",
    "धोखा|बकवास|फालतू|परेशान|शिकायत|कॉल मत|फ़ोन मत|फोन मत|मत करो|बंद करो",
)


def _wrap(cue: str) -> str:
    # Word-boundary only on the Latin side of the fragment; Devanagari has no \b semantics
    # we can rely on with combining marks.
    return rf"(?<![a-z])(?:{cue})(?![a-z])" if cue.isascii() else f"(?:{cue})"


_OBJECTION_RES: dict[Objection, re.Pattern[str]] = {
    obj: re.compile("|".join(_wrap(c) for c in cues)) for obj, cues in _OBJECTION_CUES.items()
}
_NEGATIVE_RE = re.compile("|".join(_wrap(c) for c in _NEGATIVE_CUES))

OBJECTIONS: tuple[Objection, ...] = get_args(Objection)


def detect_objections(texts: Iterable[str]) -> list[Objection]:
    """Objections raised across customer turns, de-duplicated in first-seen order."""
    found: list[Objection] = []
    for text in texts:
        t = text.lower()
        # Specific objections, in the order they appear within the turn.
        positions = {o: m.start() for o, p in _OBJECTION_RES.items() if (m := p.search(t))}
        hits: list[Objection] = sorted(positions, key=positions.__getitem__)
        if not hits and _NEGATIVE_RE.search(t):
            hits = ["other"]
        for obj in hits:
            if obj not in found:
                found.append(obj)
    return found


# --- record construction -------------------------------------------------------------------


def _disposition(t: Transcript) -> str:
    fs = t.final_state
    if fs.dnc or fs.end_reason == "dnc":
        return "dnc"
    if fs.end_reason == "wrong_person":
        return "wrong_person"
    if fs.callback_time and fs.consent:
        return "interested_callback"
    if fs.callback_time:
        return "callback_requested"
    if fs.ended:
        return "declined"
    return "incomplete"


_NEXT_ACTION = {
    "interested_callback": "executive_callback",
    "callback_requested": "retry_callback",
    "dnc": "suppress_number",
}

_OBJECTION_PHRASE: dict[Objection, str] = {
    "rate_too_high": "objected to the rate",
    "call_later": "asked to be called later",
    "not_interested": "said they are not interested",
    "already_has_loan": "mentioned an existing loan",
    "number_source": "asked how we got their number",
    "other": "pushed back",
}


def _summary(disposition: str, objections: list[Objection], callback_time: str | None) -> str:
    cb = callback_time or "an unspecified time"
    outcome = {
        "interested_callback": f"consented to an executive call; callback booked for {cb}.",
        "callback_requested": f"asked for a callback at {cb} without consenting yet.",
        "declined": "declined the offer.",
        "dnc": "asked not to be called again; number suppressed.",
        "wrong_person": "was not the intended person; call ended.",
        "incomplete": "call ended before any outcome.",
    }[disposition]
    parts = [_OBJECTION_PHRASE[o] for o in objections]
    if disposition == "incomplete" and not parts:
        text = "Call ended before any outcome."
    else:
        text = "Customer " + ", ".join([*parts, outcome])
    return redact(text)


def build_postcall_record(t: Transcript, *, created_at: float | None = None) -> PostCallRecord:
    """Deterministic CRM record for one call (no LLM). `created_at` defaults to now."""
    customer = [turn for turn in t.turns if turn.role == "customer"]
    objections = detect_objections(turn.text for turn in customer)
    trajectory = [turn.sentiment for turn in customer if turn.sentiment is not None]
    disposition = _disposition(t)
    fs = t.final_state
    return PostCallRecord(
        call_id=t.call_id,
        mode=t.mode,
        prompt_version=t.prompt_version,
        persona_id=t.persona_id,
        created_at=time.time() if created_at is None else created_at,
        disposition=disposition,
        objections=objections,
        sentiment_trajectory=trajectory,
        final_sentiment=trajectory[-1] if trajectory else None,
        consent_status="given" if fs.consent else "not_given",
        next_action=_NEXT_ACTION.get(disposition, "no_action"),
        callback_time=fs.callback_time,
        dnc=fs.dnc,
        turns=len(t.turns),
        summary=_summary(disposition, objections, fs.callback_time),
    )


# --- writers -------------------------------------------------------------------------------


def _write_json(path: Path, data: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    tmp.replace(path)


def write_postcall_records(version: str, results_dir: str = "results", fresh: bool = False) -> int:
    """Build results/<v>/postcall/<call_id>.json for every transcript; return count written."""
    layout = ResultsLayout(root=Path(results_dir), version=version)
    if not layout.transcripts.is_dir():
        return 0
    written = 0
    for src in sorted(layout.transcripts.glob("*.json")):
        transcript = Transcript.model_validate_json(src.read_text(encoding="utf-8"))
        dest = layout.postcall / f"{transcript.call_id}.json"
        if transcript.stop_reason == "error":  # crashed call: no CRM record (re-sim retries it)
            dest.unlink(missing_ok=True)
            continue
        if dest.exists() and not fresh and dest.stat().st_mtime >= src.stat().st_mtime:
            continue  # up to date (a re-simulated call has a newer transcript)
        record = build_postcall_record(transcript)
        _write_json(dest, record.model_dump(mode="json"))
        written += 1
    return written


def write_voice_records(
    transcript: Transcript, logs_dir: str | Path = "logs/calls"
) -> tuple[Path, Path]:
    """Write <call_id>.transcript.json and <call_id>.postcall.json at the end of a voice call."""
    base = Path(logs_dir)
    t_path = base / f"{transcript.call_id}.transcript.json"
    p_path = base / f"{transcript.call_id}.postcall.json"
    # Transcripts from brain.transcript are already redacted; redacting again is idempotent
    # and protects against callers that built the Transcript by hand.
    clean = Transcript.model_validate(redact_obj(transcript.model_dump(mode="json")))
    _write_json(t_path, clean.model_dump(mode="json"))
    _write_json(p_path, build_postcall_record(clean).model_dump(mode="json"))
    return t_path, p_path


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="python -m brain.postcall", description=__doc__)
    parser.add_argument("--version", default=None, help="prompt version (default: config)")
    parser.add_argument("--results", default=None, help="results dir (default: config)")
    parser.add_argument("--fresh", action="store_true", help="overwrite existing records")
    args = parser.parse_args(argv)
    version, results = args.version, args.results
    if version is None or results is None:
        from brain.config import load_config

        cfg = load_config(prompt_version=version)
        version = version or cfg.brain.prompt_version
        results = results or cfg.results_dir
    n = write_postcall_records(version, results, fresh=args.fresh)
    print(f"postcall: wrote {n} record(s) to {Path(results) / version / 'postcall'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

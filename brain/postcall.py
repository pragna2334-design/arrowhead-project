"""Post-call intelligence: turn a finished Transcript into a CRM-ready PostCallRecord.

Deterministic and offline (keyword rules, no LLM), so it costs nothing and can be re-run.
Objection ids come from the transcript's scenario pack (scenario.yaml objections[].id/cues).

    python -m brain.postcall [--scenario personal_loan] [--version v1] [--results results]
                             [--fresh]
"""

from __future__ import annotations

import argparse
import json
import re
import sys
import time
from collections.abc import Iterable
from functools import lru_cache
from pathlib import Path

from brain.redact import redact, redact_obj
from brain.scenario import DEFAULT_SCENARIO, load_scenario
from schemas import PostCallRecord, ResultsLayout, ScenarioConfig, Transcript
from schemas.postcall import Objection

# --- objection detection -------------------------------------------------------------------

# Built-in, hand-tuned regex cues per objection id. They apply to every scenario that defines
# an objection with that id (merged with that scenario's own literal `cues`), so generic ids
# (call_later, not_interested, number_source) get the same quality everywhere.
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


def _literal(cue: str) -> str:
    """A scenario cue is a plain phrase: escape it and let any whitespace run match."""
    return r"\s+".join(re.escape(w) for w in cue.lower().split())


_NEGATIVE_RE = re.compile("|".join(_wrap(c) for c in _NEGATIVE_CUES))
_SCENARIO_ID = re.compile(r"[a-z0-9_]+")

# The original personal-loan ids, kept for callers that imported this name.
OBJECTIONS: tuple[Objection, ...] = tuple(_OBJECTION_CUES)

_Plays = tuple[tuple[str, tuple[str, ...]], ...]  # ((objection id, cues), ...)
_Patterns = tuple[tuple[Objection, re.Pattern[str]], ...]


@lru_cache(maxsize=32)
def _compile_plays(plays: _Plays) -> _Patterns:
    out = []
    for obj, cues in plays:
        frags = [*_OBJECTION_CUES.get(obj, ()), *(_literal(c) for c in cues if c.strip())]
        if frags:
            out.append((obj, re.compile("|".join(_wrap(f) for f in frags))))
    return tuple(out)


def _plays_of(sc: ScenarioConfig) -> _Plays:
    return tuple((o.id, tuple(o.cues)) for o in sc.objections)


# Unknown/unloadable scenario: fall back to the built-in personal-loan detectors.
_BUILTIN_PLAYS: _Plays = tuple((o, ()) for o in _OBJECTION_CUES)


def _resolve(scenario: str | ScenarioConfig | None) -> ScenarioConfig | None:
    if isinstance(scenario, ScenarioConfig):
        return scenario
    sid = scenario or DEFAULT_SCENARIO
    if not _SCENARIO_ID.fullmatch(sid):  # ids come from transcripts: no path tricks
        return None
    try:
        return load_scenario(sid)
    except (FileNotFoundError, ValueError):
        return None


def objection_patterns(scenario: str | ScenarioConfig | None = None) -> _Patterns:
    """(objection id, compiled cue regex) in the scenario's playbook order."""
    sc = _resolve(scenario)
    return _compile_plays(_plays_of(sc) if sc is not None else _BUILTIN_PLAYS)


def detect_objections(
    texts: Iterable[str], scenario: str | ScenarioConfig | None = None
) -> list[Objection]:
    """Objections raised across customer turns, de-duplicated in first-seen order.

    Ids come from the scenario's objection playbook (default: personal_loan); clearly negative
    pushback that matches none of them is "other"."""
    patterns = objection_patterns(scenario)
    found: list[Objection] = []
    for text in texts:
        t = text.lower()
        # Specific objections, in the order they appear within the turn.
        positions = {o: m.start() for o, p in patterns if (m := p.search(t))}
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
    "premium_too_high": "objected to the premium",
    "has_employer_cover": "mentioned employer health cover",
    "bad_claim_experience": "mentioned a bad claim experience",
    "annual_fee_high": "objected to the annual fee",
    "already_have_card": "mentioned an existing card",
    "debt_concern": "raised a debt concern",
    "other": "pushed back",
}


def _phrase(objection: Objection) -> str:
    return _OBJECTION_PHRASE.get(objection) or f"raised '{objection.replace('_', ' ')}'"


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
    parts = [_phrase(o) for o in objections]
    if disposition == "incomplete" and not parts:
        text = "Call ended before any outcome."
    else:
        text = "Customer " + ", ".join([*parts, outcome])
    return redact(text)


def build_postcall_record(t: Transcript, *, created_at: float | None = None) -> PostCallRecord:
    """Deterministic CRM record for one call (no LLM). `created_at` defaults to now."""
    customer = [turn for turn in t.turns if turn.role == "customer"]
    objections = detect_objections((turn.text for turn in customer), t.scenario)
    trajectory = [turn.sentiment for turn in customer if turn.sentiment is not None]
    disposition = _disposition(t)
    fs = t.final_state
    return PostCallRecord(
        call_id=t.call_id,
        mode=t.mode,
        prompt_version=t.prompt_version,
        persona_id=t.persona_id,
        scenario=t.scenario,
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


def write_postcall_records(
    version: str,
    results_dir: str | Path = "results",
    fresh: bool = False,
    scenario: str | None = None,
) -> int:
    """Build results/<scenario>/<v>/postcall/<call_id>.json for every transcript; return count
    written. `scenario` defaults to personal_loan."""
    layout = ResultsLayout(
        root=Path(results_dir), version=version, scenario=scenario or DEFAULT_SCENARIO
    )
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
    parser.add_argument("--scenario", default=None, help="scenario id (default: env/config)")
    parser.add_argument("--version", default=None, help="prompt version (default: config)")
    parser.add_argument("--results", default=None, help="results dir (default: config)")
    parser.add_argument("--fresh", action="store_true", help="overwrite existing records")
    args = parser.parse_args(argv)
    from brain.config import load_config

    try:
        cfg = load_config(prompt_version=args.version, scenario=args.scenario)
    except (FileNotFoundError, ValueError) as e:
        print(f"bad config: {e}", file=sys.stderr)
        return 2
    version = args.version or cfg.brain.prompt_version
    results = args.results or cfg.results_dir
    scenario = cfg.scenario.id
    n = write_postcall_records(version, results, fresh=args.fresh, scenario=scenario)
    out = ResultsLayout(Path(results), version, scenario=scenario).postcall
    print(f"postcall: wrote {n} record(s) to {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

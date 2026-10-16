"""Clause chunker: decides when streamed LLM text is released to TTS (pure, no Pipecat).

Pipecat's default releases text at sentence ends. For a Hinglish voice agent the first sentence
is often long ("Haan ji, main samajh sakti hoon ki rate zyada lag raha hai, lekin ..."), so the
caller waits for the whole sentence before any audio. The clause chunker releases text at clause
boundaries instead:

    , ; : . ? ! … । ॥   and a spaced dash " — " / " – "

Rules
- A boundary is confirmed by the character after the punctuation run: whitespace confirms it;
  for the strong marks (? ! । ॥) any non-punctuation character also confirms it. Anything else
  cancels it, so numbers and times stay whole: "11.5", "5,00,000", "18:00", "Rs.500".
- A trailing run of punctuation / closing quotes stays with its clause ("Achha...", 'haan?"').
- "." after a known abbreviation (Mr, Dr, Rs, ...) or a single letter (initials, "i.e.") is not
  a boundary.
- Minimum length: the first clause of a response is released once it has >= first_min_chars
  characters (so "Haan ji," goes out immediately but "Ji," waits for more); later clauses need
  >= min_chars, so the rest of the reply is not cut into choppy fragments. A clause below the
  minimum simply stays buffered and joins the next one.
- flush() returns whatever is left (end of the LLM response) and starts a new response.

Also home of the text-aggregation mode names shared by the voice bot and the bench.
"""

from __future__ import annotations

import re

AGGREGATION_MODES = ("sentence", "clause", "token")
DEFAULT_AGGREGATION = "clause"

FIRST_MIN_CHARS = 6  # "Achha," / "Haan ji," are released at once; "Ji," / "Haan," are not
MIN_CHARS = 20  # later clauses: keep prosody natural, latency no longer matters much

_STRONG = frozenset("?!।॥")  # boundary even without a following space
_WEAK = frozenset(",;:.…")  # boundary only when followed by whitespace
_BOUNDARY = _STRONG | _WEAK
_CLOSERS = frozenset("\"'”’)]»")  # stay attached to the clause they close
_DASHES = (" — ", " – ")
_ABBREVIATIONS = frozenset(
    {"mr", "mrs", "ms", "dr", "rs", "no", "st", "vs", "sr", "jr", "ltd", "pvt", "approx", "co"}
)
_LAST_WORD = re.compile(r"([A-Za-z]+)$")


def parse_aggregation_mode(value: object) -> str:
    """Validate a `text_aggregation` config value (case-insensitive)."""
    mode = str(value).strip().lower() if value is not None else ""
    if mode not in AGGREGATION_MODES:
        raise ValueError(f"text_aggregation must be one of {AGGREGATION_MODES}, got {value!r}")
    return mode


def _is_abbreviation(before_dot: str) -> bool:
    m = _LAST_WORD.search(before_dot)
    if not m:
        return False
    word = m.group(1)
    return len(word) == 1 or word.lower() in _ABBREVIATIONS


class ClauseChunker:
    """Feed streamed text with feed(); it returns the clauses that are ready to speak."""

    def __init__(self, *, first_min_chars: int = FIRST_MIN_CHARS, min_chars: int = MIN_CHARS):
        if first_min_chars < 1 or min_chars < 1:
            raise ValueError("min chars must be >= 1")
        self.first_min_chars = first_min_chars
        self.min_chars = min_chars
        self.reset()

    def reset(self) -> None:
        """Drop buffered text (interruption) and start a new response."""
        self._buf = ""
        self._pending: int | None = None  # buffer index just after a candidate boundary
        self._pending_strong = False
        self._released_any = False

    @property
    def pending_text(self) -> str:
        """Text buffered but not yet released."""
        return self._buf.strip()

    def feed(self, text: str) -> list[str]:
        out: list[str] = []
        for ch in text:
            clause = self._push_char(ch)
            if clause:
                out.append(clause)
        return out

    def flush(self) -> str | None:
        """End of the response: return the remainder (if any) and reset."""
        rest = self._buf.strip()
        self.reset()
        return rest or None

    def _push_char(self, ch: str) -> str | None:
        self._buf += ch
        if self._pending is not None:
            if ch in _BOUNDARY or ch in _CLOSERS:
                # punctuation run ("...", "?!") or closing quote: extend the candidate
                self._pending = len(self._buf)
                self._pending_strong = self._pending_strong or ch in _STRONG
                return None
            confirmed = ch.isspace() or self._pending_strong
            at = self._pending
            self._pending, self._pending_strong = None, False
            if confirmed:
                return self._release(at)
            return None  # "11.5", "5,00,000", "18:00": not a boundary
        if ch in _BOUNDARY:
            if ch == "." and _is_abbreviation(self._buf[:-1]):
                return None
            self._pending = len(self._buf)
            self._pending_strong = ch in _STRONG
            return None
        if ch == " " and self._buf.endswith(_DASHES):
            return self._release(len(self._buf))
        return None

    def _release(self, at: int) -> str | None:
        clause = self._buf[:at].strip()
        need = self.min_chars if self._released_any else self.first_min_chars
        if len(clause) < need:
            return None  # too short: keep buffering, it joins the next clause
        self._buf = self._buf[at:]
        self._released_any = True
        return clause


def chunk_text(text: str, **kwargs: int) -> list[str]:
    """Split a complete text the way a stream of it would be released (for tests/debugging)."""
    chunker = ClauseChunker(**kwargs)
    out = chunker.feed(text)
    rest = chunker.flush()
    return [*out, rest] if rest else out

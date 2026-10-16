"""Clause chunker (pure) and its Pipecat aggregator / TTS integration, offline."""

import asyncio
import dataclasses

import pytest
from pipecat.frames.frames import LLMTextFrame

from voice.chunking import ClauseChunker, chunk_text, parse_aggregation_mode


def stream(text: str, step: int = 3, **kw) -> list[str]:
    """Feed text in small pieces (like LLM tokens) and collect releases + the flush."""
    c = ClauseChunker(**kw)
    out: list[str] = []
    for i in range(0, len(text), step):
        out += c.feed(text[i : i + step])
    rest = c.flush()
    return [*out, rest] if rest else out


@pytest.mark.parametrize("step", [1, 2, 3, 7, 1000])
def test_hinglish_clauses_independent_of_token_size(step):
    text = "Haan ji, main samajh sakti hoon. Rate 11.5% se shuru hota hai, aur amount 5,00,000 tak."
    assert stream(text, step) == [
        "Haan ji,",
        "main samajh sakti hoon.",
        "Rate 11.5% se shuru hota hai,",
        "aur amount 5,00,000 tak.",
    ]


def test_ack_released_as_soon_as_next_char_arrives():
    c = ClauseChunker()
    assert c.feed("Haan ji,") == []  # waits one char: could still be "5,00"
    assert c.feed(" ") == ["Haan ji,"]
    assert c.pending_text == ""


def test_numbers_and_times_never_split():
    for text in (
        "Aapka loan 11.5 percent par milega aur ye offer achha hai.",
        "Amount 5,00,000 tak ho sakta hai aur tenure 60 mahine.",
        "Main aapko kal 18:00 baje call karungi aur details bhejungi.",
        "Processing fee Rs.500 hai aur koi hidden charge nahi hai.",
    ):
        assert chunk_text(text) == [text]


def test_min_length_merges_tiny_fragments():
    # "Ji," is below first_min_chars: it joins the next clause
    assert chunk_text("Ji, bilkul. Main details bhejti hoon aapko abhi.") == [
        "Ji, bilkul.",
        "Main details bhejti hoon aapko abhi.",
    ]
    # later clauses need min_chars (20): short ones are merged into longer chunks
    assert chunk_text("Achha, haan, theek, sahi hai, chaliye aage badhte hain.") == [
        "Achha,",
        "haan, theek, sahi hai,",
        "chaliye aage badhte hain.",
    ]
    assert chunk_text("Haan, ok.", first_min_chars=3, min_chars=3) == ["Haan,", "ok."]


def test_devanagari_danda_and_double_danda():
    text = "हाँ जी। आपका loan approve ho sakta hai॥ धन्यवाद"
    assert chunk_text(text) == ["हाँ जी।", "आपका loan approve ho sakta hai॥", "धन्यवाद"]
    # strong marks confirm even without a following space
    assert chunk_text("Theek hai।Aage batati hoon.") == ["Theek hai।", "Aage batati hoon."]


def test_punctuation_runs_quotes_and_abbreviations():
    assert chunk_text("Achha... toh aap Mr. Sharma ji hain? Ji haan bolo.") == [
        "Achha...",
        "toh aap Mr. Sharma ji hain?",
        "Ji haan bolo.",
    ]
    assert chunk_text('Aapne kaha "nahi chahiye," theek hai, main samajh gayi.') == [
        'Aapne kaha "nahi chahiye,"',
        "theek hai, main samajh gayi.",
    ]
    assert chunk_text("Dr. A. K. Verma ne bola tha ki rate kam hai.") == [
        "Dr. A. K. Verma ne bola tha ki rate kam hai."
    ]


def test_spaced_dash_is_a_boundary():
    assert chunk_text("Rate thoda zyada lag raha hai — lekin EMI kam ho sakti hai.") == [
        "Rate thoda zyada lag raha hai —",
        "lekin EMI kam ho sakti hai.",
    ]
    assert chunk_text("EMI-free period nahi hai.") == ["EMI-free period nahi hai."]


def test_flush_returns_rest_and_starts_new_response():
    c = ClauseChunker()
    assert c.feed("Haan ji, aapka callback book") == ["Haan ji,"]
    assert c.flush() == "aapka callback book"
    assert c.flush() is None
    # new response: the small first-clause threshold applies again
    assert c.feed("Achha, ") == ["Achha,"]
    c.reset()
    assert c.pending_text == "" and c.flush() is None


def test_question_and_exclamation():
    assert chunk_text("Kya aap abhi baat kar sakte hain? Do minute lagenge!") == [
        "Kya aap abhi baat kar sakte hain?",
        "Do minute lagenge!",
    ]


def test_parse_aggregation_mode():
    assert parse_aggregation_mode("Clause") == "clause"
    for bad in ("words", "", None, 3):
        with pytest.raises(ValueError):
            parse_aggregation_mode(bad)
    with pytest.raises(ValueError):
        ClauseChunker(first_min_chars=0)


# --- Pipecat glue -----------------------------------------------------------------------------


def test_clause_text_aggregator_contract():
    from pipecat.utils.text.base_text_aggregator import AggregationType

    from voice.aggregation import ClauseTextAggregator

    async def go():
        agg = ClauseTextAggregator()
        got = [a async for a in agg.aggregate("Haan ji, main 11.5 bol rahi")]
        assert [a.text for a in got] == ["Haan ji,"]
        assert got[0].type == AggregationType.SENTENCE
        assert agg.text.text == "main 11.5 bol rahi"
        await agg.handle_interruption()
        assert agg.text.text == "" and await agg.flush() is None
        _ = [a async for a in agg.aggregate("bas itna")]
        assert (await agg.flush()).text == "bas itna"
        await agg.reset()

    asyncio.run(go())


@pytest.fixture
def bot(monkeypatch):
    for k in ("DEEPGRAM_API_KEY", "SARVAM_API_KEY", "DEEPSEEK_API_KEY", "ELEVENLABS_API_KEY"):
        monkeypatch.setenv(k, "dummy")
    import voice.bot

    return voice.bot


def test_sarvam_tts_releases_clauses_from_llm_tokens(bot, cfg):
    """End to end inside the real SarvamTTSService: LLM tokens in, clauses to synthesis out."""
    tts = bot.build_tts(cfg)  # config default: clause
    assert type(tts._text_aggregator).__name__ == "ClauseTextAggregator"
    sent: list[str] = []

    async def record(frame, *a, **k):
        sent.append(frame.text)

    tts._push_tts_frames = record

    async def go():
        toks = ["Haan", " ji", ",", " rate", " 11", ".", "5", "%", " se", " shuru", " hai", "."]
        for tok in [*toks, " Aur"]:
            await tts._process_text_frame(LLMTextFrame(tok))
        assert sent == ["Haan ji,", "rate 11.5% se shuru hai."]
        assert (await tts._text_aggregator.flush()).text == "Aur"

    asyncio.run(go())


def test_install_guard_rejects_token_mode(bot, cfg):
    from voice.aggregation import install_clause_aggregator

    c = dataclasses.replace(cfg, tts={**cfg.tts, "sarvam": {**cfg.tts["sarvam"]}})
    c.tts["sarvam"]["text_aggregation"] = "token"
    tts = bot.build_tts(c)
    with pytest.raises(RuntimeError):
        install_clause_aggregator(tts)
    with pytest.raises(TypeError):
        install_clause_aggregator(object())

from brain.prompt import OPENING_CUE
from voice.context import context_to_turns, tag_last_user_message


def msgs():
    return [
        {"role": "system", "content": "sys"},
        {"role": "user", "content": OPENING_CUE},
        {"role": "assistant", "content": "Namaste, main Priya, Vriddhi Finance ki AI assistant."},
        {"role": "user", "content": "kya matlab?"},
        {"role": "assistant", "content": "", "tool_calls": [{"id": "x"}]},
        {"role": "tool", "content": "{}", "tool_call_id": "x"},
    ]


def test_tags_newest_user_message_once():
    m = msgs()
    assert tag_last_user_message(m) == "confused"
    assert m[3]["content"] == "kya matlab?\n[customer_mood: confused]"
    assert tag_last_user_message(m) == "confused"  # idempotent
    assert m[3]["content"].count("customer_mood") == 1


def test_opening_cue_not_tagged():
    m = msgs()[:3]
    assert tag_last_user_message(m) is None and m[1]["content"] == OPENING_CUE


def test_multipart_content():
    m = [{"role": "user", "content": [{"type": "text", "text": "aur batao"}]}]
    assert tag_last_user_message(m) == "interested"


def test_context_to_turns_strips_tags_cue_and_tools():
    m = msgs()
    tag_last_user_message(m)
    turns = context_to_turns(m, started_at=100.0)
    assert [(t.role, t.text, t.sentiment) for t in turns] == [
        ("agent", "Namaste, main Priya, Vriddhi Finance ki AI assistant.", None),
        ("customer", "kya matlab?", "confused"),
    ]
    assert turns[1].ts == 101.0

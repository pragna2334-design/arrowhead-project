from brain.state import CallState
from brain.tools import TOOL_SPECS, closing_line, execute_tool, in_callback_window, parse_hhmm


def st():
    return CallState(call_id="t", customer_name="Rahul Verma")


def test_specs_are_well_formed():
    names = {s.name for s in TOOL_SPECS}
    assert names == {"record_consent", "book_callback", "end_call", "mark_do_not_call"}
    for s in TOOL_SPECS:
        assert set(s.required) <= set(s.properties)


def test_consent_and_callback_inside_hours():
    s = st()
    assert execute_tool(s, "record_consent", {})["ok"]
    r = execute_tool(
        s, "book_callback", {"day": "tomorrow", "time": "18:00"}, callback_hours=(10, 19)
    )
    assert r == {"ok": True, "callback_time": "tomorrow 18:00"}
    assert s.consent is True and s.callback_time == "tomorrow 18:00"


def test_callback_outside_hours_rejected():
    s = st()
    r = execute_tool(s, "book_callback", {"day": "today", "time": "21:00"}, callback_hours=(10, 19))
    assert not r["ok"] and "outside" in r["error"] and s.callback_time is None
    # end hour is exclusive, start inclusive
    assert in_callback_window(10, 0, (10, 19)) and not in_callback_window(19, 0, (10, 19))


def test_callback_bad_format_rejected():
    s = st()
    assert not execute_tool(s, "book_callback", {"day": "kal", "time": "6 pm"})["ok"]
    assert parse_hhmm("6 pm") is None and parse_hhmm("09:30") == (9, 30)


def test_dnc_ends_call_and_blocks_further_tools():
    s = st()
    execute_tool(s, "mark_do_not_call", {})
    assert s.dnc and s.ended and s.end_reason == "dnc"
    assert not execute_tool(s, "record_consent", {})["ok"]
    assert s.consent is None
    assert "hata diya" in closing_line(s)


def test_end_call_reason_and_unknown_reason():
    s = st()
    execute_tool(s, "end_call", {"reason": "nonsense"})
    assert s.end_reason == "other" and s.ended
    s2 = st()
    execute_tool(s2, "end_call", {"reason": "declined"})
    assert "Koi baat nahi" in closing_line(s2)


def test_closing_line_mentions_callback_time():
    s = st()
    execute_tool(s, "book_callback", {"day": "Saturday", "time": "11:30"})
    execute_tool(s, "end_call", {"reason": "callback_booked"})
    assert "Saturday 11:30" in closing_line(s)


def test_tool_calls_are_logged_with_turn_index():
    s = st()
    execute_tool(s, "unknown_tool", {})
    assert s.tool_calls[-1].name == "unknown_tool" and not s.tool_calls[-1].result["ok"]
    assert s.tool_calls[-1].turn_index == -1

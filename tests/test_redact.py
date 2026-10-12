"""Offline tests for brain.redact."""

from __future__ import annotations

import pytest
from loguru import logger

from brain.redact import install_log_redaction, redact, redact_obj


@pytest.mark.parametrize(
    "text",
    [
        "9876543210",
        "98765 43210",
        "98765-43210",
        "+91-98765-43210",
        "+91 98765 43210",
        "+919876543210",
        "0091 9876543210",
        "91 9876543210",
        "09876543210",
        "987-654-3210",
        "9876 543 210",
        "6123456789",
    ],
)
def test_phone_formats(text: str) -> None:
    assert redact(f"mera number {text} hai") == "mera number [PHONE] hai"


@pytest.mark.parametrize("text", ["ABCDE1234F", "abcde1234f", "AbCdE1234f"])
def test_pan(text: str) -> None:
    assert redact(f"PAN: {text}.") == "PAN: [PAN]."


@pytest.mark.parametrize("text", ["234567890123", "2345 6789 0123", "2345-6789-0123"])
def test_aadhaar(text: str) -> None:
    assert redact(f"aadhaar {text} hai") == "aadhaar [AADHAAR] hai"


def test_aadhaar_not_starting_0_or_1() -> None:
    # 12 digits starting with 1 is not an Aadhaar (and not a phone either).
    assert redact("id 123456789012") == "id 123456789012"


def test_aadhaar_before_phone() -> None:
    # A 4-4-4 Aadhaar must not be partially masked as a phone.
    assert redact("9876 5432 1012") == "[AADHAAR]"


@pytest.mark.parametrize(
    "text", ["rahul.verma@gmail.com", "a_b+tag@mail.co.in", "RAHUL99@Example.ORG"]
)
def test_email(text: str) -> None:
    assert redact(f"mail karo {text} pe") == "mail karo [EMAIL] pe"


def test_email_containing_digits_is_one_token() -> None:
    assert redact("rahul9876543210@gmail.com") == "[EMAIL]"


def test_devanagari_digits() -> None:
    assert redact("मेरा नंबर ९८७६५ ४३२१० है") == "मेरा नंबर [PHONE] है"
    assert redact("आधार २३४५ ६७८९ ०१२३") == "आधार [AADHAAR]"
    # Non-PII Devanagari digits stay exactly as written.
    assert redact("कल शाम ६ बजे, ५ लाख") == "कल शाम ६ बजे, ५ लाख"


def test_mixed_sentence() -> None:
    text = "Main Rahul, 98765 43210, PAN abcde1234f, mail r@x.in, aadhaar 2345 6789 0123."
    assert redact(text) == "Main Rahul, [PHONE], PAN [PAN], mail [EMAIL], aadhaar [AADHAAR]."


@pytest.mark.parametrize(
    "text",
    [
        "Rs 5,00,000 tak",
        "500000",
        "₹3,00,000",
        "5 lakh",
        "Rs 5 lakh tak, 12 se 60 mahine",
        "11.5% se 18% per annum",
        "18:00",
        "tomorrow 18:00",
        "Kal shaam 6 baje",
        "12-10-2026",
        "2026-10-12",
        "12 to 60 months",
        "EMI 10,624 per month",
        "call_id fx000001",
        "1791800000.0",
        "Rs 1,00,00,000",
    ],
)
def test_no_false_positives(text: str) -> None:
    assert redact(text) == text


@pytest.mark.parametrize(
    "text",
    [
        "number 98765 43210 aur PAN ABCDE1234F, mail a@b.com, aadhaar 2345 6789 0123",
        "+91-98765-43210",
        "९८७६५४३२१०",
        "nothing here",
        "",
    ],
)
def test_idempotent(text: str) -> None:
    once = redact(text)
    assert redact(once) == once


def test_redact_obj_recursive() -> None:
    obj = {
        "9876543210": "call 9876543210",  # keys untouched
        "nested": [{"pan": "ABCDE1234F"}, ("a@b.com", 9876543210, None)],
        "n": 5,
        "ok": True,
    }
    out = redact_obj(obj)
    assert out == {
        "9876543210": "call [PHONE]",
        "nested": [{"pan": "[PAN]"}, ("[EMAIL]", 9876543210, None)],
        "n": 5,
        "ok": True,
    }
    assert isinstance(out["nested"][1], tuple)
    assert obj["9876543210"] == "call 9876543210"  # input not mutated


def test_log_redaction() -> None:
    lines: list[str] = []
    sink_id = logger.add(lines.append, format="{message}")
    try:
        install_log_redaction()
        install_log_redaction()  # safe to call twice
        logger.info("customer said {} and PAN {}", "98765 43210", "abcde1234f")
        logger.bind(call="x").warning("mail rahul@gmail.com")
    finally:
        logger.remove(sink_id)
    assert lines == ["customer said [PHONE] and PAN [PAN]\n", "mail [EMAIL]\n"]

"""PII redaction. PLACEHOLDER: replaced by the post-call/redaction work package.

Frozen interface:
    redact(text: str) -> str          # masks phone, PAN, Aadhaar, email
    redact_obj(obj: Any) -> Any       # recursive over dict/list/tuple/str
"""

from __future__ import annotations

from typing import Any


def redact(text: str) -> str:
    return text


def redact_obj(obj: Any) -> Any:
    return obj

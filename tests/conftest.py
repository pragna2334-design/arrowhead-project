"""Shared offline test helpers: a stub OpenAI-compatible client (no network)."""

from __future__ import annotations

import copy
import json
from types import SimpleNamespace as NS

import pytest

from brain.config import load_config


def chat_msg(text: str | None = None, calls: list[tuple[str, dict]] | tuple = ()):
    """One chat-completion response with optional tool calls."""
    tcs = [
        NS(id=f"call{i}", function=NS(name=n, arguments=json.dumps(a)))
        for i, (n, a) in enumerate(calls)
    ]
    return NS(choices=[NS(message=NS(content=text, tool_calls=tcs or None))])


class StubCompletions:
    def __init__(self, script):
        self.script = list(script)
        self.requests: list[dict] = []

    async def create(self, **kw):
        self.requests.append(copy.deepcopy(kw))  # snapshot: callers mutate lists later
        return self.script.pop(0)


def stub_client(script):
    comp = StubCompletions(script)
    return NS(chat=NS(completions=comp)), comp


@pytest.fixture
def cfg():
    return load_config()

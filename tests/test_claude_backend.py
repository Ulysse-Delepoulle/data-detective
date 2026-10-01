"""Tests for the Claude backend. Fast: no network, no key, no spend.

A FakeClient mimics the anthropic client's shape (client.messages.create
returning a message whose .content is a list of blocks). We inject it so we
can check the request we build and the way we parse the response, without
touching the real API.

The live test at the end is double-gated: it needs both RUN_LIVE_CLAUDE=1 and
a key, so a normal test run never makes a paid call.
"""
import os
from types import SimpleNamespace

import pytest

from datadetective.llm.claude_backend import ClaudeBackend, _text_from_message


def _text_block(text: str) -> SimpleNamespace:
    return SimpleNamespace(type="text", text=text)


class FakeMessages:
    def __init__(self, blocks: list) -> None:
        self.blocks = blocks
        self.calls: list[dict] = []

    def create(self, **kwargs):
        self.calls.append(kwargs)
        return SimpleNamespace(content=self.blocks)


class FakeClient:
    def __init__(self, blocks: list) -> None:
        self.messages = FakeMessages(blocks)


def test_generate_returns_block_text():
    client = FakeClient([_text_block("hello")])
    backend = ClaudeBackend(client=client)

    assert backend.generate("hi") == "hello"


def test_request_has_model_tokens_and_message():
    client = FakeClient([_text_block("x")])
    backend = ClaudeBackend(model="claude-sonnet-5", max_tokens=123, client=client)

    backend.generate("my prompt")
    sent = client.messages.calls[0]

    assert sent["model"] == "claude-sonnet-5"
    assert sent["max_tokens"] == 123
    assert sent["messages"] == [{"role": "user", "content": "my prompt"}]


def test_system_included_only_when_given():
    client = FakeClient([_text_block("x")])
    backend = ClaudeBackend(client=client)

    backend.generate("p", system="you are a bot")
    assert client.messages.calls[0]["system"] == "you are a bot"

    backend.generate("p")
    assert "system" not in client.messages.calls[1]


def test_multiple_text_blocks_are_joined():
    client = FakeClient([_text_block("foo"), _text_block("bar")])
    backend = ClaudeBackend(client=client)

    assert backend.generate("p") == "foobar"


def test_non_text_blocks_are_ignored():
    blocks = [
        SimpleNamespace(type="thinking", text="secret"),
        _text_block("visible"),
    ]
    assert _text_from_message(SimpleNamespace(content=blocks)) == "visible"


@pytest.mark.integration
def test_live_claude_smoke():
    if os.environ.get("RUN_LIVE_CLAUDE") != "1":
        pytest.skip("set RUN_LIVE_CLAUDE=1 to run the live Claude call")
    if not os.environ.get("ANTHROPIC_API_KEY"):
        pytest.skip("no ANTHROPIC_API_KEY set")

    backend = ClaudeBackend(max_tokens=16)
    out = backend.generate("Reply with the single word: pong")
    assert isinstance(out, str) and out.strip()

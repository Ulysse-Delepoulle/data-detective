"""Claude API implementation of the LLMBackend contract.

This is the Phase 5 payoff of the base class. The agent, graph, nodes, report
builder, and cache do not change at all: they call backend.generate(...) and
do not care that the text now comes from Anthropic's API instead of the local
Ollama server.

Two Claude specifics the adapter hides from the rest of the app:
- max_tokens is required by the API, so we always send it.
- the response is a list of content blocks, not a single string, so we
  concatenate the text blocks and return one string.

The API key is never passed in code. The anthropic client reads it from the
ANTHROPIC_API_KEY environment variable, loaded from a .env file in dev.
"""
from __future__ import annotations

from typing import Any

from .base import LLMBackend

# The cheapest current model, a sensible default for development and for
# staying inside the budget. Override per run for higher-quality final runs.
DEFAULT_MODEL = "claude-haiku-4-5"


def _text_from_message(message: Any) -> str:
    """Join the text of every text block in a Claude response."""
    parts = [
        block.text
        for block in message.content
        if getattr(block, "type", None) == "text"
    ]
    return "".join(parts)


class ClaudeBackend(LLMBackend):
    def __init__(
        self,
        model: str = DEFAULT_MODEL,
        max_tokens: int = 2048,
        api_key: str | None = None,
        client: Any | None = None,
    ) -> None:
        self.model = model
        self.max_tokens = max_tokens
        if client is not None:
            # Dependency injection: tests pass a fake client so no network or
            # key is needed, and no API package import is required.
            self.client = client
        else:
            # Imported here, not at module top, so the module loads and the
            # tests run even where the anthropic package is absent.
            import anthropic

            self.client = anthropic.Anthropic(api_key=api_key)

    def generate(self, prompt: str, system: str | None = None) -> str:
        kwargs: dict[str, Any] = {
            "model": self.model,
            "max_tokens": self.max_tokens,
            "messages": [{"role": "user", "content": prompt}],
        }
        if system is not None:
            kwargs["system"] = system
        message = self.client.messages.create(**kwargs)
        return _text_from_message(message)

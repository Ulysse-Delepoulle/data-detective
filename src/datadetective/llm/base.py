"""The contract every LLM backend must follow.

This is an abstract base class. It declares the method that all backends
(Ollama now, Claude later) must provide, without saying how they do it.
The rest of the code depends on this interface, not on any one provider,
which is what lets us swap the local model for Claude by changing a single
line instead of rewriting the callers.
"""
from __future__ import annotations

from abc import ABC, abstractmethod


class LLMBackend(ABC):
    """Interface for anything that can turn a prompt into text."""

    @abstractmethod
    def generate(self, prompt: str, system: str | None = None) -> str:
        """Send a prompt and optional system instruction, return the reply.

        Args:
            prompt: the request to send to the model.
            system: optional instruction that sets the model's role or rules.

        Returns:
            The model's text response as a plain string.
        """
        raise NotImplementedError

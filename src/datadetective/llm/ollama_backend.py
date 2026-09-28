"""LLM backend that talks to a local Ollama server.

Ollama runs a web server on this machine (default http://localhost:11434).
This backend sends the prompt to that server's /api/generate endpoint and
returns the model's text. It implements the LLMBackend contract, so it can
be swapped for the Claude backend later without changing any calling code.
"""
from __future__ import annotations

import requests

from .base import LLMBackend


class OllamaBackend(LLMBackend):
    def __init__(
        self,
        model: str = "qwen2.5:7b",
        host: str = "http://localhost:11434",
        timeout: int = 180,
    ) -> None:
        self.model = model
        self.host = host.rstrip("/")
        # Seconds to wait for a reply. Local inference on CPU can be slow,
        # so this is generous.
        self.timeout = timeout

    def generate(self, prompt: str, system: str | None = None) -> str:
        payload = {
            "model": self.model,
            "prompt": prompt,
            "stream": False,  # return one full response, not token by token
        }
        if system is not None:
            payload["system"] = system

        response = requests.post(
            f"{self.host}/api/generate",
            json=payload,
            timeout=self.timeout,
        )
        # Turn any HTTP error (bad model name, server down) into an exception
        # instead of silently returning garbage.
        response.raise_for_status()
        return response.json()["response"]

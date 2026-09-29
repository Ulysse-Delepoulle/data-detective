"""A caching wrapper around any LLM backend.

CachingBackend wraps another LLMBackend and implements the same contract,
so callers cannot tell the difference. It hashes each request into a key,
looks for that key on disk, and returns the stored answer on a hit. Only
on a miss does it call the wrapped backend and save the result.

Why: model calls are slow (local CPU) and cost money (Claude later).
Identical requests should be answered once. The cache persists on disk so
it survives across runs, which speeds up development and protects the
budget.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

from .base import LLMBackend


class CachingBackend(LLMBackend):
    def __init__(
        self,
        inner: LLMBackend,
        cache_dir: str | Path = ".cache/llm",
        namespace: str | None = None,
    ) -> None:
        self.inner = inner
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        # The namespace keeps different models' answers apart, since the same
        # prompt can give different results on different models.
        self.namespace = namespace or getattr(
            inner, "model", inner.__class__.__name__
        )
        # Simple counters so we can see how well the cache is working.
        self.hits = 0
        self.misses = 0

    def _key(self, prompt: str, system: str | None) -> str:
        # A stable string built from everything that affects the answer,
        # then hashed. sort_keys makes the JSON deterministic.
        raw = json.dumps(
            {"namespace": self.namespace, "system": system, "prompt": prompt},
            sort_keys=True,
        )
        return hashlib.sha256(raw.encode("utf-8")).hexdigest()

    def _path(self, key: str) -> Path:
        return self.cache_dir / f"{key}.json"

    def generate(self, prompt: str, system: str | None = None) -> str:
        path = self._path(self._key(prompt, system))
        if path.exists():
            self.hits += 1
            return json.loads(path.read_text(encoding="utf-8"))["response"]

        self.misses += 1
        response = self.inner.generate(prompt, system=system)
        path.write_text(
            json.dumps(
                {
                    "namespace": self.namespace,
                    "system": system,
                    "prompt": prompt,
                    "response": response,
                },
                ensure_ascii=False,
                indent=2,
            ),
            encoding="utf-8",
        )
        return response

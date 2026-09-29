"""Tests for the CachingBackend. Fast: no model, no Docker.

A CountingBackend records how many times the real work ran, so we can prove
the cache actually prevented calls."""
from datadetective.llm.base import LLMBackend
from datadetective.llm.cache import CachingBackend


class CountingBackend(LLMBackend):
    def __init__(self) -> None:
        self.calls = 0

    def generate(self, prompt: str, system: str | None = None) -> str:
        self.calls += 1
        return f"response-{self.calls}"


def test_repeat_request_is_served_from_cache(tmp_path):
    inner = CountingBackend()
    cached = CachingBackend(inner, cache_dir=tmp_path)

    first = cached.generate("hello")
    second = cached.generate("hello")

    assert inner.calls == 1        # the second request did not reach inner
    assert first == second
    assert cached.hits == 1
    assert cached.misses == 1


def test_different_prompt_is_a_miss(tmp_path):
    inner = CountingBackend()
    cached = CachingBackend(inner, cache_dir=tmp_path)

    cached.generate("one")
    cached.generate("two")

    assert inner.calls == 2
    assert cached.misses == 2


def test_system_prompt_is_part_of_the_key(tmp_path):
    inner = CountingBackend()
    cached = CachingBackend(inner, cache_dir=tmp_path)

    cached.generate("q", system="role A")
    cached.generate("q", system="role B")

    # Same prompt, different system text, so these are different requests.
    assert inner.calls == 2


def test_cache_persists_across_instances(tmp_path):
    inner1 = CountingBackend()
    first = CachingBackend(inner1, cache_dir=tmp_path).generate("hello")

    # A brand new backend and wrapper pointing at the same cache folder.
    inner2 = CountingBackend()
    second = CachingBackend(inner2, cache_dir=tmp_path).generate("hello")

    assert inner2.calls == 0       # answer came from the file written before
    assert first == second

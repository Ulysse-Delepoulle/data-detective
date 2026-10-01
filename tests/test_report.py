"""Tests for the Phase 4 report builder. Fast: no model, no Docker.

A ScriptedReportBackend returns a fixed sequence of replies, so we can force
the parse/validation failures the retry loop is meant to recover from, and
prove it recovers deterministically.
"""
import json

from datadetective.llm.base import LLMBackend
from datadetective.report.builder import _extract_json, build_report

QUESTION = "Total revenue by region."
OUTPUT = "West    95.5\nSouth   86.5\nNorth   72.5"

VALID_REPORT = {
    "question": QUESTION,
    "finding": "The West region has the highest revenue.",
    "supporting_numbers": ["West: 95.5", "South: 86.5", "North: 72.5"],
    "method": "Summed units times price grouped by region.",
    "caveats": ["Only ten rows in the sample."],
}


class ScriptedReportBackend(LLMBackend):
    """Returns a fixed list of replies in order."""

    def __init__(self, replies: list[str]) -> None:
        self.replies = replies
        self.calls = 0

    def generate(self, prompt: str, system: str | None = None) -> str:
        reply = self.replies[min(self.calls, len(self.replies) - 1)]
        self.calls += 1
        return reply


def test_valid_json_builds_report_on_first_try():
    backend = ScriptedReportBackend([json.dumps(VALID_REPORT)])
    report = build_report(QUESTION, OUTPUT, backend=backend)

    assert backend.calls == 1
    assert report is not None
    assert report.finding == VALID_REPORT["finding"]
    assert report.supporting_numbers == VALID_REPORT["supporting_numbers"]


def test_retries_after_broken_json_then_succeeds():
    backend = ScriptedReportBackend(["not json at all", json.dumps(VALID_REPORT)])
    report = build_report(QUESTION, OUTPUT, backend=backend)

    assert backend.calls == 2        # first reply failed to parse
    assert report is not None
    assert report.finding == VALID_REPORT["finding"]


def test_retries_after_validation_error_then_succeeds():
    missing_field = {k: v for k, v in VALID_REPORT.items() if k != "method"}
    backend = ScriptedReportBackend(
        [json.dumps(missing_field), json.dumps(VALID_REPORT)]
    )
    report = build_report(QUESTION, OUTPUT, backend=backend)

    assert backend.calls == 2        # first reply was valid JSON but missing a field
    assert report is not None
    assert report.method == VALID_REPORT["method"]


def test_empty_supporting_numbers_triggers_retry():
    empty_numbers = {**VALID_REPORT, "supporting_numbers": []}
    backend = ScriptedReportBackend(
        [json.dumps(empty_numbers), json.dumps(VALID_REPORT)]
    )
    report = build_report(QUESTION, OUTPUT, backend=backend)

    assert backend.calls == 2        # empty list failed validation, so it retried
    assert report is not None
    assert report.supporting_numbers == VALID_REPORT["supporting_numbers"]


def test_returns_none_when_all_attempts_fail():
    backend = ScriptedReportBackend(["still not json"])
    report = build_report(QUESTION, OUTPUT, backend=backend, max_attempts=3)

    assert backend.calls == 3        # tried the maximum number of times
    assert report is None


def test_chart_files_come_from_our_code_not_the_model():
    # The model puts a bogus chart in its reply; our real list must win.
    with_bogus_chart = {**VALID_REPORT, "chart_files": ["hallucinated.png"]}
    backend = ScriptedReportBackend([json.dumps(with_bogus_chart)])
    report = build_report(
        QUESTION, OUTPUT, backend=backend, chart_files=["/out/real_chart.png"]
    )

    assert report is not None
    assert report.chart_files == ["/out/real_chart.png"]


def test_extract_json_handles_fence_and_surrounding_text():
    fenced = "Here is your report:\n```json\n{\"a\": 1}\n```\nThanks!"
    assert _extract_json(fenced) == '{"a": 1}'

    prose = 'Sure. {"a": 1} Let me know if you need more.'
    assert _extract_json(prose) == '{"a": 1}'

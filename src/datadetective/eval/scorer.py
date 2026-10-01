"""Deterministic scoring of a report against a benchmark case.

A case passes when every expected token appears somewhere in the agent's
output: the report's finding, method, and supporting numbers, plus the raw
stdout. Matching is case-insensitive and ignores commas, so numeric answers
like "650000" match whether the agent printed "650000" or "650,000".

This never calls a model, so scores are free and reproducible.
"""
from __future__ import annotations

from dataclasses import dataclass
from statistics import mean
from typing import Optional

from ..report.models import AnalysisReport
from .cases import BenchmarkCase


def _normalize(text: str) -> str:
    return text.lower().replace(",", "")


def _haystack(report: Optional[AnalysisReport], answer: str) -> str:
    parts: list[str] = [answer]
    if report is not None:
        parts.append(report.finding)
        parts.append(report.method)
        parts.extend(report.supporting_numbers)
    return _normalize(" ".join(parts))


def case_passed(
    case: BenchmarkCase, report: Optional[AnalysisReport], answer: str
) -> bool:
    text = _haystack(report, answer)
    return all(_normalize(token) in text for token in case.expect)


@dataclass
class CaseResult:
    case: BenchmarkCase
    passed: bool
    has_report: bool
    attempts: int
    duration: float


def summarize(results: list[CaseResult]) -> dict:
    """Aggregate per-case results into headline metrics."""
    if not results:
        return {"cases": 0, "accuracy": 0.0, "completion": 0.0, "avg_attempts": 0.0}
    return {
        "cases": len(results),
        "accuracy": sum(r.passed for r in results) / len(results),
        "completion": sum(r.has_report for r in results) / len(results),
        "avg_attempts": mean(r.attempts for r in results),
    }


def summarize_by_tier(results: list[CaseResult]) -> dict[str, dict]:
    """Metrics per tier, so smoke and benchmark are reported separately."""
    tiers = sorted({r.case.tier for r in results})
    return {tier: summarize([r for r in results if r.case.tier == tier]) for tier in tiers}

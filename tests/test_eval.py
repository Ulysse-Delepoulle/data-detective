"""Tests for the eval scorer. Fast: no agent, no model, no Docker.

These check the scoring logic itself, not the agent's quality (that is what the
full eval run measures).
"""
from datadetective.eval.cases import CASES, BenchmarkCase
from datadetective.eval.scorer import (
    CaseResult,
    case_passed,
    summarize,
    summarize_by_tier,
)
from datadetective.report.models import AnalysisReport


def _report(finding="", numbers=(), method="") -> AnalysisReport:
    return AnalysisReport(
        question="q",
        finding=finding,
        supporting_numbers=list(numbers) or ["n/a"],
        method=method or "m",
        caveats=[],
    )


CASE = BenchmarkCase("c", "d.csv", "q", ("West",))


def test_pass_when_token_in_finding():
    report = _report(finding="The West region leads.")
    assert case_passed(CASE, report, answer="") is True


def test_fail_when_token_absent():
    report = _report(finding="The North region leads.")
    assert case_passed(CASE, report, answer="") is False


def test_token_can_match_in_stdout_answer():
    report = _report(finding="unrelated")
    assert case_passed(CASE, report, answer="West 95.5") is True


def test_number_match_ignores_commas():
    case = BenchmarkCase("c", "d.csv", "q", ("650000",))
    report = _report(numbers=["Total salary: 650,000"])
    assert case_passed(case, report, answer="") is True


def test_no_report_fails_but_stdout_still_checked():
    case = BenchmarkCase("c", "d.csv", "q", ("Gadget",))
    assert case_passed(case, None, answer="Gadget wins") is True
    assert case_passed(case, None, answer="nothing here") is False


def test_summarize_computes_rates():
    results = [
        CaseResult(CASE, passed=True, has_report=True, attempts=1, duration=0.1),
        CaseResult(CASE, passed=False, has_report=True, attempts=2, duration=0.2),
    ]
    summary = summarize(results)
    assert summary["cases"] == 2
    assert summary["accuracy"] == 0.5
    assert summary["completion"] == 1.0
    assert summary["avg_attempts"] == 1.5


def test_benchmark_has_enough_cases():
    # A mix of synthetic and real-dataset cases; at least a dozen.
    assert len(CASES) >= 12


def test_both_tiers_present():
    tiers = {c.tier for c in CASES}
    assert "smoke" in tiers and "benchmark" in tiers


def test_summarize_by_tier_splits_groups():
    smoke_case = BenchmarkCase("s", "d.csv", "q", ("West",), tier="smoke")
    bench_case = BenchmarkCase("b", "d.csv", "q", ("West",), tier="benchmark")
    results = [
        CaseResult(smoke_case, passed=True, has_report=True, attempts=1, duration=0.1),
        CaseResult(bench_case, passed=False, has_report=True, attempts=1, duration=0.1),
    ]
    by_tier = summarize_by_tier(results)
    assert by_tier["smoke"]["accuracy"] == 1.0
    assert by_tier["benchmark"]["accuracy"] == 0.0

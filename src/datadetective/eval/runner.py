"""Run the agent over the benchmark and print metrics.

Usage:
    python -m datadetective.eval.runner                 # local Ollama (free)
    python -m datadetective.eval.runner --backend cloud # Claude (costs money)
    python -m datadetective.eval.runner --no-cache      # skip the cache

The agent runs end to end per case (sandbox plus model), so this takes a few
minutes on the local model. Results are cached, so a re-run is fast.
"""
from __future__ import annotations

import argparse
import shutil
import time
from pathlib import Path

from ..agent.graph import run_agent
from ..llm.base import LLMBackend
from ..llm.cache import CachingBackend
from .cases import CASES, BenchmarkCase
from .scorer import CaseResult, case_passed, summarize, summarize_by_tier

_DATA_DIR = Path(__file__).resolve().parents[3] / "data" / "sample_datasets"


def _dataset_path(case: BenchmarkCase) -> str:
    return str(_DATA_DIR / case.dataset)


def run_eval(
    backend: LLMBackend,
    cases: tuple[BenchmarkCase, ...] = CASES,
    max_attempts: int = 3,
) -> list[CaseResult]:
    results: list[CaseResult] = []
    for case in cases:
        start = time.monotonic()
        final = run_agent(
            case.question,
            _dataset_path(case),
            backend=backend,
            max_attempts=max_attempts,
        )
        duration = time.monotonic() - start

        report = final.get("report")
        answer = final.get("answer", "")
        results.append(
            CaseResult(
                case=case,
                passed=case_passed(case, report, answer),
                has_report=report is not None,
                attempts=final.get("attempts", 0),
                duration=duration,
            )
        )

        # Each run leaves a sandbox temp folder on disk; clean it up.
        execution = final.get("execution")
        if execution is not None and execution.work_dir:
            shutil.rmtree(execution.work_dir, ignore_errors=True)

    return results


def _print_table(results: list[CaseResult]) -> None:
    name_width = max(len(r.case.name) for r in results)
    print(f"{'case'.ljust(name_width)}  tier       result  attempts   time")
    print("-" * (name_width + 38))
    for r in results:
        mark = "PASS" if r.passed else "FAIL"
        print(
            f"{r.case.name.ljust(name_width)}  {r.case.tier:<9}  {mark:>4}  "
            f"{r.attempts:>8}  {r.duration:>5.1f}s"
        )


def _print_summary_line(label: str, s: dict) -> None:
    print(
        f"{label:<12} accuracy: {s['accuracy']:.0%}   "
        f"completion: {s['completion']:.0%}   "
        f"avg attempts: {s['avg_attempts']:.2f}   "
        f"cases: {s['cases']}"
    )


def main() -> None:
    # Load .env so the cloud backend finds ANTHROPIC_API_KEY when run as a CLI.
    from dotenv import load_dotenv

    load_dotenv()

    parser = argparse.ArgumentParser(description="Run the DataDetective eval.")
    parser.add_argument("--backend", choices=["local", "cloud"], default="local")
    parser.add_argument(
        "--model",
        default=None,
        help="Override the model id (e.g. claude-sonnet-4-6 for cloud).",
    )
    parser.add_argument("--max-attempts", type=int, default=3)
    parser.add_argument("--no-cache", action="store_true", help="bypass the LLM cache")
    args = parser.parse_args()

    # Imported here so a missing anthropic package does not break local runs.
    if args.backend == "cloud":
        from ..llm.claude_backend import ClaudeBackend

        inner: LLMBackend = (
            ClaudeBackend(model=args.model) if args.model else ClaudeBackend()
        )
    else:
        from ..llm.ollama_backend import OllamaBackend

        inner = OllamaBackend(model=args.model) if args.model else OllamaBackend()

    backend = inner if args.no_cache else CachingBackend(inner)

    model_label = getattr(inner, "model", args.backend)
    print(f"Running {len(CASES)} cases on backend={args.backend} ({model_label}) ...\n")
    results = run_eval(backend, max_attempts=args.max_attempts)

    print()
    _print_table(results)

    print()
    by_tier = summarize_by_tier(results)
    for tier, s in by_tier.items():
        _print_summary_line(tier, s)
    _print_summary_line("overall", summarize(results))


if __name__ == "__main__":
    main()

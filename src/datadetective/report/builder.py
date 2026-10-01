"""Turn a successful analysis into a validated AnalysisReport.

This is the same self-correction idea as the Phase 3 code loop, but the
failure signal is different. In Phase 3 the signal was a non-zero exit code
from the sandbox. Here the signal is a parsing or validation error:

    ask the model for a JSON report
    try to parse the JSON and validate it against AnalysisReport
        success -> return the report
        failure -> feed the model its bad output and the exact error,
                   ask again, up to max_attempts

The retry stays inside this one function instead of being extra graph nodes,
because the whole loop is model-side (no sandbox), so it is a single concern.
"""
from __future__ import annotations

import json
import re

from pydantic import ValidationError

from ..llm.base import LLMBackend
from .models import AnalysisReport

REPORT_SYSTEM_PROMPT = (
    "You are a data analyst writing a structured report. You are given a "
    "business question and the raw output of analysis code that already "
    "answered it. Return your report as a single JSON object with exactly "
    "these keys:\n"
    '- "question": string, the original question.\n'
    '- "finding": string, a one-sentence headline answer.\n'
    '- "supporting_numbers": list of strings, the key figures behind the '
    "finding, for example \"West region revenue: 95.5\". Include at least one "
    "number; never leave this empty. Use only numbers that actually appear in "
    "the analysis output below. Copy them exactly and do not invent, round, or "
    "reformat any figure.\n"
    '- "method": string, how the answer was computed (which columns and '
    "what aggregation).\n"
    '- "caveats": list of strings, assumptions or limitations. Use an empty '
    "list if there are none.\n"
    "Reply with only the JSON object and nothing else."
)


def _build_report_prompt(question: str, analysis_output: str) -> str:
    return (
        f"Question: {question}\n\n"
        f"Analysis output:\n{analysis_output}\n\n"
        "Write the JSON report."
    )


def _extract_json(reply: str) -> str:
    """Pull the JSON object out of a reply.

    Models often wrap JSON in a code fence or add a sentence around it, so we
    strip a fence if present, then fall back to the span from the first "{"
    to the last "}".
    """
    fence = re.search(r"```(?:json)?\s*(.*?)```", reply, re.DOTALL)
    text = fence.group(1).strip() if fence else reply.strip()
    start = text.find("{")
    end = text.rfind("}")
    if start != -1 and end != -1 and end > start:
        return text[start : end + 1]
    return text


def build_report(
    question: str,
    analysis_output: str,
    backend: LLMBackend,
    chart_files: list[str] | None = None,
    max_attempts: int = 3,
) -> AnalysisReport | None:
    """Generate a validated report, retrying on parse or validation errors.

    Returns the report on success, or None if every attempt failed.
    """
    base_prompt = _build_report_prompt(question, analysis_output)
    last_error = ""

    for _ in range(max_attempts):
        prompt = base_prompt
        if last_error:
            prompt += (
                "\n\nYour previous reply was not accepted. The error was:\n"
                f"{last_error}\n"
                "Return a corrected JSON object that fixes this."
            )

        reply = backend.generate(prompt, system=REPORT_SYSTEM_PROMPT)
        raw = _extract_json(reply)

        try:
            data = json.loads(raw)
        except json.JSONDecodeError as exc:
            last_error = f"The reply was not valid JSON: {exc}"
            continue

        try:
            report = AnalysisReport.model_validate(data)
        except ValidationError as exc:
            last_error = str(exc)
            continue

        # Success. Fill chart_files from the real sandbox outputs rather than
        # trusting whatever the model may have put there.
        if chart_files:
            report.chart_files = list(chart_files)
        return report

    return None

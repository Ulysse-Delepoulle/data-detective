"""The shared state that flows through the agent graph.

This is the agent's entire memory. Every node reads from it and returns
the fields it wants to change; LangGraph merges those changes back in.
If a step needs to know something, it lives here, not in hidden variables.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, TypedDict

from ..report.models import AnalysisReport
from ..sandbox.executor import ExecutionResult


@dataclass
class AttemptRecord:
    """One code attempt, kept so the UI can show the full reasoning trail."""

    attempt: int        # 1-based attempt number
    code: str           # the code the model wrote on this attempt
    stdout: str
    stderr: str
    exit_code: int
    timed_out: bool


class AgentState(TypedDict):
    # Inputs, set once when the run starts.
    question: str
    dataset_path: str
    max_attempts: int

    # Working fields, updated as the graph runs.
    code: str                          # the latest code the model wrote
    execution: Optional[ExecutionResult]  # the latest sandbox result
    attempts: int                      # how many code attempts so far
    history: list[AttemptRecord]       # every attempt, in order

    # Output, filled in at the end.
    answer: str                        # raw stdout, or a failure message
    report: Optional[AnalysisReport]   # the structured deliverable, if built

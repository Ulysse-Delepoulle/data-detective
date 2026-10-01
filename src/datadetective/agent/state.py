"""The shared state that flows through the agent graph.

This is the agent's entire memory. Every node reads from it and returns
the fields it wants to change; LangGraph merges those changes back in.
If a step needs to know something, it lives here, not in hidden variables.
"""
from __future__ import annotations

from typing import Optional, TypedDict

from ..report.models import AnalysisReport
from ..sandbox.executor import ExecutionResult


class AgentState(TypedDict):
    # Inputs, set once when the run starts.
    question: str
    dataset_path: str
    max_attempts: int

    # Working fields, updated as the graph runs.
    code: str                          # the latest code the model wrote
    execution: Optional[ExecutionResult]  # the latest sandbox result
    attempts: int                      # how many code attempts so far

    # Output, filled in at the end.
    answer: str                        # raw stdout, or a failure message
    report: Optional[AnalysisReport]   # the structured deliverable, if built

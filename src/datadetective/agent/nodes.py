"""The nodes (steps) of the agent graph, plus the routing decision.

Each node is a function that takes the current state and returns only the
fields it changes. The nodes reuse the Phase 2 pieces (prompt, code
extraction, sandbox) and add self-correction: a failed attempt is fed back
to the model as an error to fix.

Nodes are built by small factory functions (make_*_node) so each can
capture the backend or sandbox config it needs while still presenting the
plain "state -> updates" shape that LangGraph calls.
"""
from __future__ import annotations

from typing import Callable

from ..llm.base import LLMBackend
from ..report.builder import build_report
from ..sandbox.config import DEFAULT_CONFIG, SandboxConfig
from ..sandbox.executor import ExecutionResult, run_code
from .single_step import SYSTEM_PROMPT, build_prompt, extract_code
from .state import AgentState

# File extensions we treat as charts worth listing in the report.
_IMAGE_EXTENSIONS = (".png", ".jpg", ".jpeg", ".svg")


def _prompt_for_attempt(state: AgentState) -> str:
    """First attempt: the normal prompt. Retry: also include the previous
    code and its error so the model can fix its own mistake."""
    prompt = build_prompt(state["question"], state["dataset_path"])
    execution = state.get("execution")
    if execution is not None and execution.exit_code != 0:
        prompt += (
            "\n\nYour previous code failed. Here is the code you wrote:\n"
            f"```python\n{state['code']}\n```\n"
            f"It produced this error:\n{execution.stderr.strip()}\n"
            "Fix the problem and return the corrected code."
        )
    return prompt


def make_write_code_node(backend: LLMBackend) -> Callable[[AgentState], dict]:
    def write_code(state: AgentState) -> dict:
        prompt = _prompt_for_attempt(state)
        reply = backend.generate(prompt, system=SYSTEM_PROMPT)
        code = extract_code(reply)
        return {"code": code, "attempts": state["attempts"] + 1}

    return write_code


def make_execute_node(
    config: SandboxConfig = DEFAULT_CONFIG,
) -> Callable[[AgentState], dict]:
    def execute(state: AgentState) -> dict:
        result = run_code(
            state["code"],
            dataset_path=state["dataset_path"],
            config=config,
        )
        return {"execution": result}

    return execute


def _chart_files(execution: ExecutionResult) -> list[str]:
    """Pick the image files out of everything the sandbox wrote."""
    return [
        path
        for path in execution.output_files
        if path.lower().endswith(_IMAGE_EXTENSIONS)
    ]


def make_report_node(
    backend: LLMBackend, max_attempts: int = 3
) -> Callable[[AgentState], dict]:
    """Build a node that turns a successful run into a structured report.

    Only reached after the code succeeded, so execution is present and clean.
    """

    def write_report(state: AgentState) -> dict:
        execution = state["execution"]
        report = build_report(
            state["question"],
            execution.stdout.strip(),
            backend=backend,
            chart_files=_chart_files(execution),
            max_attempts=max_attempts,
        )
        return {"report": report}

    return write_report


def finalize(state: AgentState) -> dict:
    """Fill in the final answer from the last execution, whether it worked
    or we ran out of attempts."""
    execution = state.get("execution")
    if execution is not None and execution.exit_code == 0 and not execution.timed_out:
        return {"answer": execution.stdout.strip()}
    return {
        "answer": (
            "Failed to produce a working analysis after "
            f"{state['attempts']} attempt(s)."
        )
    }


def decide_next(state: AgentState) -> str:
    """Conditional routing after execute.

    Returns one of the labels the graph maps to a next node:
    - "success": the code ran cleanly, go finalize.
    - "retry":   it failed but attempts remain, write code again.
    - "give_up": it failed and no attempts remain, go finalize.
    """
    execution = state.get("execution")
    succeeded = (
        execution is not None
        and execution.exit_code == 0
        and not execution.timed_out
    )
    if succeeded:
        return "success"
    if state["attempts"] >= state["max_attempts"]:
        return "give_up"
    return "retry"

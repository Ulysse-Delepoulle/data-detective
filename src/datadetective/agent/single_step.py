"""Phase 2 single-step flow: question in, answer out.

There is no planning and no error-retry here yet (those are Phase 3). This
does exactly one pass: build a prompt from the question and a preview of the
dataset, ask the model to write Python, pull the code out of the reply, and
run that code in the Phase 1 sandbox.
"""
from __future__ import annotations

import csv
import re
from dataclasses import dataclass
from pathlib import Path

from ..llm.base import LLMBackend
from ..sandbox.config import DEFAULT_CONFIG, SandboxConfig
from ..sandbox.executor import ExecutionResult, run_code

# The system prompt sets the model's role and the rules it must follow.
SYSTEM_PROMPT = (
    "You are an expert data analyst. You answer questions by writing Python "
    "code that uses pandas. Rules:\n"
    "- The dataset is a CSV file. Load it with exactly these lines:\n"
    "    import os\n"
    "    import pandas as pd\n"
    "    df = pd.read_csv(os.environ['DATASET_PATH'])\n"
    "- You may use pandas, numpy, matplotlib, and seaborn only.\n"
    "- Print the final answer clearly using print().\n"
    "- Reply with exactly one Python code block and nothing else."
)


@dataclass
class SingleStepResult:
    """Everything produced by one pass, so we can inspect each stage."""

    question: str
    model_reply: str
    code: str
    execution: ExecutionResult


def _dataset_preview(dataset_path: str, sample_rows: int = 3) -> str:
    """Read the header and a few rows with the stdlib csv module.

    We avoid pandas on the host on purpose: the host stays light and the
    heavy data libraries live only inside the sandbox.
    """
    path = Path(dataset_path)
    lines: list[str] = []
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.reader(handle)
        header = next(reader, [])
        lines.append("Columns: " + ", ".join(header))
        lines.append("Sample rows:")
        for index, row in enumerate(reader):
            if index >= sample_rows:
                break
            lines.append("  " + ", ".join(row))
    return "\n".join(lines)


def build_prompt(question: str, dataset_path: str) -> str:
    """Combine a preview of the dataset with the user's question."""
    preview = _dataset_preview(dataset_path)
    return (
        f"{preview}\n\n"
        f"Question: {question}\n\n"
        "Write Python code to answer the question."
    )


def extract_code(reply: str) -> str:
    """Pull the Python out of a reply that uses ``` fences.

    Falls back to the whole reply if no fenced code block is found.
    """
    match = re.search(r"```(?:python)?\s*(.*?)```", reply, re.DOTALL)
    if match:
        return match.group(1).strip()
    return reply.strip()


def answer_question(
    question: str,
    dataset_path: str,
    backend: LLMBackend,
    config: SandboxConfig = DEFAULT_CONFIG,
) -> SingleStepResult:
    """Run the full single-step flow and return every stage's output."""
    prompt = build_prompt(question, dataset_path)
    reply = backend.generate(prompt, system=SYSTEM_PROMPT)
    code = extract_code(reply)
    execution = run_code(code, dataset_path=dataset_path, config=config)
    return SingleStepResult(
        question=question,
        model_reply=reply,
        code=code,
        execution=execution,
    )

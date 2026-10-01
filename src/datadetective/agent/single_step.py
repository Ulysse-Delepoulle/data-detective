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
    "- Also print the key numbers behind the answer, so the figures are "
    "visible in the output, not just the conclusion.\n"
    "- Reply with exactly one Python code block and nothing else."
)

# Appended to the system prompt only for questions that ask for a chart, so
# non-visual questions keep the leaner prompt and a higher first-try hit rate.
_PLOT_RULE = (
    "\n- This question asks for a visualization. Create it with matplotlib and "
    "save it to the current directory with "
    "plt.savefig('chart.png', bbox_inches='tight'). Do not call plt.show(). "
    "Make the chart readable:\n"
    "    - set a wide figure, for example plt.figure(figsize=(11, 5));\n"
    "    - add a title and label both axes;\n"
    "    - if the x-axis has many or long labels, rotate them with "
    "plt.xticks(rotation=45, ha='right') and show at most about 12 ticks by "
    "thinning them (for example set_xticks on a subset), so they do not "
    "overlap;\n"
    "    - call plt.tight_layout() before saving.\n"
    "  Always also print the exact data series behind the chart, so the real "
    "numbers appear in the output."
)

_VIZ_KEYWORDS = (
    "plot", "chart", "graph", "visual", "trend", "distribution",
    "histogram", "over time", "evolution", "evolution of",
)


def wants_visualization(question: str) -> bool:
    q = question.lower()
    return any(keyword in q for keyword in _VIZ_KEYWORDS)


def build_system_prompt(question: str) -> str:
    """The base rules, plus the plotting rule only when a chart is requested."""
    return SYSTEM_PROMPT + (_PLOT_RULE if wants_visualization(question) else "")


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
    reply = backend.generate(prompt, system=build_system_prompt(question))
    code = extract_code(reply)
    execution = run_code(code, dataset_path=dataset_path, config=config)
    return SingleStepResult(
        question=question,
        model_reply=reply,
        code=code,
        execution=execution,
    )

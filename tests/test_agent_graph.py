"""Tests for the Phase 3 self-correcting agent graph.

Two groups:
1. Pure routing logic (decide_next): fast, no Docker.
2. The retry loop end to end, driven by a ScriptedBackend that returns a
   fixed sequence of replies. This proves recovery deterministically
   instead of hoping the real model fails. Needs Docker.
"""
import shutil

import pytest

from datadetective.agent.graph import run_agent
from datadetective.agent.nodes import decide_next
from datadetective.llm.base import LLMBackend
from datadetective.sandbox.executor import ExecutionResult

GOOD_CODE = (
    "import os\n"
    "import pandas as pd\n"
    "df = pd.read_csv(os.environ['DATASET_PATH'])\n"
    "df['revenue'] = df['units'] * df['price']\n"
    "print(df.groupby('region')['revenue'].sum().sort_values(ascending=False))\n"
)
BAD_CODE = "print(this_name_does_not_exist)\n"

# A valid JSON report, used as the scripted reply for the write_report node
# after the code succeeds. The backend wraps every reply in a fence, which
# the report builder strips before parsing.
REPORT_JSON = (
    '{"question": "Total revenue by region.", '
    '"finding": "The West region has the highest revenue.", '
    '"supporting_numbers": ["West: 95.5"], '
    '"method": "Summed units times price grouped by region.", '
    '"caveats": []}'
)

DATASET = "data/sample_datasets/sample_sales.csv"
QUESTION_OK = "Total revenue by region."


# --------------------------------------------------------------------------
# 1. Pure routing logic: no Docker.
# --------------------------------------------------------------------------

def _result(exit_code: int, timed_out: bool = False) -> ExecutionResult:
    return ExecutionResult(
        stdout="", stderr="", exit_code=exit_code,
        timed_out=timed_out, duration_seconds=0.1,
    )


def test_decide_success():
    state = {"execution": _result(0), "attempts": 1, "max_attempts": 3}
    assert decide_next(state) == "success"


def test_decide_retry_when_attempts_remain():
    state = {"execution": _result(1), "attempts": 1, "max_attempts": 3}
    assert decide_next(state) == "retry"


def test_decide_give_up_when_no_attempts_remain():
    state = {"execution": _result(1), "attempts": 3, "max_attempts": 3}
    assert decide_next(state) == "give_up"


def test_decide_timeout_is_not_success():
    state = {"execution": _result(0, timed_out=True), "attempts": 1, "max_attempts": 3}
    assert decide_next(state) == "retry"


# --------------------------------------------------------------------------
# 2. Retry loop end to end with a scripted backend. Needs Docker.
# --------------------------------------------------------------------------

class ScriptedBackend(LLMBackend):
    """Returns a fixed list of replies in order, so the loop is deterministic.

    Each reply is wrapped in a code fence, as a real model would."""

    def __init__(self, code_sequence: list[str]) -> None:
        self.code_sequence = code_sequence
        self.calls = 0

    def generate(self, prompt: str, system: str | None = None) -> str:
        code = self.code_sequence[min(self.calls, len(self.code_sequence) - 1)]
        self.calls += 1
        return f"```python\n{code}\n```"


@pytest.mark.docker
def test_retry_then_success():
    # Replies in order: bad code, good code, then the JSON report that the
    # write_report node asks for once the code has succeeded.
    backend = ScriptedBackend([BAD_CODE, GOOD_CODE, REPORT_JSON])
    final = run_agent(QUESTION_OK, DATASET, backend=backend, max_attempts=3)
    try:
        assert backend.calls == 3          # one bad code, one good code, one report
        assert final["attempts"] == 2      # attempts counts code writes only
        assert final["execution"].exit_code == 0
        assert "West" in final["answer"]   # the good code's output
        assert final["report"] is not None
        assert "West" in final["report"].finding
    finally:
        shutil.rmtree(final["execution"].work_dir, ignore_errors=True)


@pytest.mark.docker
def test_gives_up_after_max_attempts():
    backend = ScriptedBackend([BAD_CODE])   # always returns failing code
    final = run_agent(QUESTION_OK, DATASET, backend=backend, max_attempts=2)
    try:
        assert final["attempts"] == 2
        assert final["execution"].exit_code != 0
        assert final["answer"].startswith("Failed to produce")
    finally:
        shutil.rmtree(final["execution"].work_dir, ignore_errors=True)

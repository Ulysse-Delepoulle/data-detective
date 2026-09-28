"""Tests for the Phase 2 single-step flow.

Three groups:
1. Pure logic (extract_code, preview, prompt): fast, no LLM, no Docker.
2. Wiring test: a fake backend feeds fixed code through the real pipeline
   and into the sandbox. Deterministic. Needs Docker.
3. Live test: hits the real Ollama model. Lenient and skipped if Ollama
   is not running.
"""
import shutil
import subprocess

import pytest
import requests

from datadetective.agent.single_step import (
    answer_question,
    build_prompt,
    extract_code,
    _dataset_preview,
)
from datadetective.llm.base import LLMBackend
from datadetective.sandbox.config import DEFAULT_CONFIG

OLLAMA_HOST = "http://localhost:11434"


# --------------------------------------------------------------------------
# 1. Pure logic: no LLM, no Docker, instant.
# --------------------------------------------------------------------------

def test_extract_code_from_python_fence():
    reply = "Sure, here it is:\n```python\nprint('hi')\n```\nThat prints hi."
    assert extract_code(reply) == "print('hi')"


def test_extract_code_from_plain_fence():
    reply = "```\nx = 1\nprint(x)\n```"
    assert extract_code(reply) == "x = 1\nprint(x)"


def test_extract_code_takes_first_block_only():
    reply = "```python\na = 1\n```\nand then\n```python\nb = 2\n```"
    assert extract_code(reply) == "a = 1"


def test_extract_code_falls_back_to_whole_reply():
    reply = "print('no fences here')"
    assert extract_code(reply) == "print('no fences here')"


@pytest.fixture
def csv_file(tmp_path):
    path = tmp_path / "d.csv"
    path.write_text(
        "region,units\nNorth,10\nSouth,5\nWest,3\nEast,1\n",
        encoding="utf-8",
    )
    return str(path)


def test_dataset_preview_lists_columns(csv_file):
    preview = _dataset_preview(csv_file)
    assert "Columns:" in preview
    assert "region" in preview
    assert "units" in preview


def test_dataset_preview_limits_sample_rows(csv_file):
    preview = _dataset_preview(csv_file, sample_rows=2)
    data_lines = [line for line in preview.splitlines() if line.startswith("  ")]
    assert len(data_lines) == 2


def test_build_prompt_includes_question_and_columns(csv_file):
    prompt = build_prompt("What is the total number of units?", csv_file)
    assert "What is the total number of units?" in prompt
    assert "region" in prompt


# --------------------------------------------------------------------------
# 2. Wiring test: fake backend + real pipeline + real sandbox. Deterministic.
# --------------------------------------------------------------------------

class FakeBackend(LLMBackend):
    """A backend that ignores the prompt and returns fixed code.

    Because it implements the LLMBackend contract, answer_question cannot
    tell it apart from the real Ollama backend. This lets us test the whole
    flow without the slowness and randomness of a real model.
    """

    def __init__(self, code: str) -> None:
        self.code = code

    def generate(self, prompt: str, system: str | None = None) -> str:
        return f"```python\n{self.code}\n```"


@pytest.mark.docker
def test_pipeline_wiring_with_fake_backend():
    code = (
        "import os\n"
        "import pandas as pd\n"
        "df = pd.read_csv(os.environ['DATASET_PATH'])\n"
        "df['revenue'] = df['units'] * df['price']\n"
        "totals = df.groupby('region')['revenue'].sum().sort_values(ascending=False)\n"
        "print(totals)\n"
    )
    result = answer_question(
        "total revenue by region",
        "data/sample_datasets/sample_sales.csv",
        backend=FakeBackend(code),
    )
    try:
        assert result.execution.exit_code == 0
        assert result.execution.timed_out is False
        # Known correct numbers for the sample dataset.
        assert "West" in result.execution.stdout
        assert "95.5" in result.execution.stdout
    finally:
        shutil.rmtree(result.execution.work_dir, ignore_errors=True)


# --------------------------------------------------------------------------
# 3. Live test: real model. Lenient and skipped when Ollama is down.
# --------------------------------------------------------------------------

def _ollama_up() -> bool:
    try:
        return requests.get(f"{OLLAMA_HOST}/api/tags", timeout=3).status_code == 200
    except requests.RequestException:
        return False


def _image_built() -> bool:
    try:
        done = subprocess.run(
            ["docker", "image", "inspect", DEFAULT_CONFIG.image],
            capture_output=True,
            text=True,
        )
        return done.returncode == 0
    except FileNotFoundError:
        return False


@pytest.mark.docker
@pytest.mark.integration
def test_live_model_writes_runnable_code():
    if not _ollama_up():
        pytest.skip("Ollama server not responding")
    if not _image_built():
        pytest.skip(f"Image {DEFAULT_CONFIG.image} not built")

    from datadetective.llm.ollama_backend import OllamaBackend

    result = answer_question(
        "What is the total revenue (units times price) for each region?",
        "data/sample_datasets/sample_sales.csv",
        backend=OllamaBackend(),
    )
    try:
        # The model output varies, so we only check that it followed the
        # instructions well enough to produce loadable, non-empty code.
        assert result.code.strip() != ""
        assert "read_csv" in result.code
        assert "DATASET_PATH" in result.code
    finally:
        shutil.rmtree(result.execution.work_dir, ignore_errors=True)

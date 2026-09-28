"""Automated proof that every sandbox isolation limit holds.

Each test runs a small piece of code through the executor and asserts
that the wall behaved. These need Docker running and the sandbox image
built (docker build -f docker/Dockerfile.sandbox -t datadetective-sandbox:latest .).
"""
import shutil
import subprocess
from dataclasses import replace

import pytest

from datadetective.sandbox.config import DEFAULT_CONFIG
from datadetective.sandbox.executor import run_code

# Mark every test in this file as needing Docker.
pytestmark = pytest.mark.docker


@pytest.fixture(scope="session", autouse=True)
def require_docker_image():
    """Skip the whole file (with a helpful message) if Docker or the image
    is not available, rather than failing in a confusing way."""
    try:
        result = subprocess.run(
            ["docker", "image", "inspect", DEFAULT_CONFIG.image],
            capture_output=True,
            text=True,
        )
    except FileNotFoundError:
        pytest.skip("Docker CLI not found on PATH")
    if result.returncode != 0:
        pytest.skip(
            f"Image {DEFAULT_CONFIG.image} not built. Build it with: "
            f"docker build -f docker/Dockerfile.sandbox "
            f"-t {DEFAULT_CONFIG.image} ."
        )


@pytest.fixture
def run(tmp_path):
    """Provide a run_code wrapper that deletes each run's temp folder after
    the test, so the suite leaves nothing behind."""
    created_dirs = []

    def _run(code, **kwargs):
        result = run_code(code, **kwargs)
        if result.work_dir:
            created_dirs.append(result.work_dir)
        return result

    yield _run

    for path in created_dirs:
        shutil.rmtree(path, ignore_errors=True)


def test_normal_output(run):
    result = run("print('hello from sandbox')")
    assert result.exit_code == 0
    assert result.timed_out is False
    assert "hello from sandbox" in result.stdout


def test_runs_as_non_root(run):
    result = run("import os; print(os.getuid())")
    assert result.exit_code == 0
    # uid 1000 is the non-root sandbox user, never 0 (root).
    assert result.stdout.strip() == "1000"


def test_network_blocked(run):
    result = run(
        "import urllib.request\n"
        "try:\n"
        "    urllib.request.urlopen('http://example.com', timeout=5)\n"
        "    print('REACHED')\n"
        "except Exception as e:\n"
        "    print('blocked', type(e).__name__)\n"
    )
    assert "REACHED" not in result.stdout
    assert "blocked" in result.stdout


def test_root_filesystem_is_read_only(run):
    result = run(
        "try:\n"
        "    open('/etc/evil', 'w').write('x')\n"
        "    print('WROTE')\n"
        "except Exception as e:\n"
        "    print('readonly', type(e).__name__)\n"
    )
    assert "WROTE" not in result.stdout
    assert "readonly" in result.stdout


def test_output_folder_is_writable(run):
    result = run("open('/out/made.txt', 'w').write('allowed'); print('ok')")
    assert result.exit_code == 0
    assert "ok" in result.stdout
    # The file the code wrote should be visible back on the host.
    assert any(p.endswith("made.txt") for p in result.output_files)


def test_memory_is_capped(run):
    # Try to allocate far more than the 512m cap; the kernel must kill it.
    result = run("x = bytearray(2_000_000_000); print('ALLOCATED', len(x))")
    assert "ALLOCATED" not in result.stdout
    assert result.exit_code != 0


def test_timeout_kills_infinite_loop(run):
    short = replace(DEFAULT_CONFIG, timeout_seconds=5)
    result = run("while True:\n    pass\n", config=short)
    assert result.timed_out is True
    # It should be killed near the 5s cap, not run forever.
    assert result.duration_seconds < 20


def test_dataset_mounted_read_only(run, tmp_path):
    # Create a tiny dataset on the host and pass it to the sandbox.
    dataset = tmp_path / "tiny.csv"
    dataset.write_text("a,b\n1,2\n3,4\n", encoding="utf-8")
    result = run(
        "import os\n"
        "path = os.environ['DATASET_PATH']\n"
        "print('rows', sum(1 for _ in open(path)))\n"
        "try:\n"
        "    open(path, 'a').write('x')\n"
        "    print('WROTE')\n"
        "except Exception as e:\n"
        "    print('dataset readonly', type(e).__name__)\n",
        dataset_path=str(dataset),
    )
    assert result.exit_code == 0
    assert "rows 3" in result.stdout      # header + 2 data lines
    assert "WROTE" not in result.stdout
    assert "dataset readonly" in result.stdout

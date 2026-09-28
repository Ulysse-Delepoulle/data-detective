"""Runs untrusted Python code inside a locked-down Docker container.

Phase 1 component: there is no LLM here. Given a string of code and an
optional dataset file, this starts a throwaway container with every
isolation limit turned on, runs the code, captures the output, and
enforces a timeout by killing the container if it runs too long.

The tunable limits live in config.py. This file is the logic that turns
those limits into "docker run" flags and manages one execution.
"""
from __future__ import annotations

import subprocess
import tempfile
import time
import uuid
from dataclasses import dataclass, field
from pathlib import Path

from .config import DEFAULT_CONFIG, SandboxConfig


@dataclass
class ExecutionResult:
    """Everything the caller needs to know about one run."""

    stdout: str
    stderr: str
    exit_code: int
    timed_out: bool
    duration_seconds: float
    # Absolute host paths of any files the code wrote to the output mount.
    output_files: list[str] = field(default_factory=list)
    # The host temp folder for this run. Left on disk so outputs survive;
    # the caller is responsible for deleting it when done.
    work_dir: str = ""


def _to_docker_path(path: Path) -> str:
    """Docker Desktop on Windows accepts forward-slash paths like D:/foo.

    A Windows path uses backslashes, but the "docker" command wants
    forward slashes in mount sources, so we convert.
    """
    return str(path).replace("\\", "/")


def run_code(
    code: str,
    dataset_path: str | None = None,
    config: SandboxConfig = DEFAULT_CONFIG,
) -> ExecutionResult:
    """Run one string of Python code in the sandbox and return the result."""

    # A unique working area on the host for this single run. It holds the
    # code we will mount read-only and an output folder we mount writable.
    run_dir = Path(tempfile.mkdtemp(prefix="ddsandbox_"))
    code_dir = run_dir / "code"
    out_dir = run_dir / "out"
    code_dir.mkdir()
    out_dir.mkdir()

    (code_dir / "code.py").write_text(code, encoding="utf-8")

    # A unique name so we can target this exact container to kill it.
    container_name = f"ddsandbox_{uuid.uuid4().hex[:12]}"

    # Build "docker run" with every isolation flag. See the table in the
    # project notes for which flag enforces which limit.
    cmd = [
        "docker", "run",
        "--rm",                                  # remove container on exit
        "--name", container_name,                # so we can kill on timeout
        "--network", "none",                     # limit 2: no network
        "--memory", config.memory,               # limit 3: memory cap
        "--memory-swap", config.memory,          # no swap beyond memory
        "--cpus", str(config.cpus),              # limit 4: cpu cap
        "--pids-limit", str(config.pids_limit),  # block fork bombs
        "--read-only",                           # limit 1: root fs read-only
        "--cap-drop", "ALL",                     # drop all Linux capabilities
        "--security-opt", "no-new-privileges",   # cannot escalate privileges
        "--user", "1000:1000",                   # limit 6: non-root
        "--tmpfs", f"/tmp:size={config.tmpfs_size}",  # writable scratch
        "--mount",
        f"type=bind,source={_to_docker_path(code_dir)},target=/sandbox,readonly",
        "--mount",
        f"type=bind,source={_to_docker_path(out_dir)},target=/out",
        "-w", "/out",                            # working dir = output mount
        "-e", "MPLCONFIGDIR=/tmp",               # matplotlib cache in tmpfs
        "-e", "PYTHONDONTWRITEBYTECODE=1",       # do not write .pyc files
        "-e", "PYTHONUNBUFFERED=1",              # flush prints immediately
    ]

    # Optionally mount the dataset read-only and tell the code where it is
    # via the DATASET_PATH environment variable.
    if dataset_path is not None:
        ds = Path(dataset_path).resolve()
        target = f"/data/{ds.name}"
        cmd += [
            "--mount",
            f"type=bind,source={_to_docker_path(ds)},target={target},readonly",
            "-e", f"DATASET_PATH={target}",
        ]

    cmd += [config.image, "python", "/sandbox/code.py"]

    start = time.monotonic()
    timed_out = False
    try:
        proc = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=config.timeout_seconds,
        )
        stdout, stderr, exit_code = proc.stdout, proc.stderr, proc.returncode
    except subprocess.TimeoutExpired as exc:
        # limit 5: the "docker run" client was killed by the timeout, but
        # the container keeps running, so we kill it explicitly by name.
        timed_out = True
        subprocess.run(
            ["docker", "kill", container_name],
            capture_output=True,
            text=True,
        )
        stdout = exc.stdout or "" if isinstance(exc.stdout, str) else (
            exc.stdout.decode() if exc.stdout else ""
        )
        stderr = exc.stderr or "" if isinstance(exc.stderr, str) else (
            exc.stderr.decode() if exc.stderr else ""
        )
        exit_code = -1

    duration = time.monotonic() - start

    output_files = [
        str(p) for p in sorted(out_dir.rglob("*")) if p.is_file()
    ]

    return ExecutionResult(
        stdout=stdout,
        stderr=stderr,
        exit_code=exit_code,
        timed_out=timed_out,
        duration_seconds=duration,
        output_files=output_files,
        work_dir=str(run_dir),
    )

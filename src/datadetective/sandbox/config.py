"""Tunable limits for the sandbox executor.

Everything adjustable about the sandbox lives here in one place, so the
limits are easy to find and change without touching the execution logic
in executor.py.
"""
from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class SandboxConfig:
    # Name of the Docker image built from docker/Dockerfile.sandbox.
    image: str = "datadetective-sandbox:latest"

    # Hard memory cap for the container. A runaway allocation is killed
    # by the kernel instead of eating the host's RAM.
    memory: str = "512m"

    # CPU cap in whole cores. 1.0 means the container may use at most
    # one core no matter how many the host has.
    cpus: float = 1.0

    # Wall-clock seconds before the container is force-killed. This is
    # what stops an infinite loop.
    timeout_seconds: int = 30

    # Maximum number of processes and threads inside the container.
    # A low cap blocks fork bombs (code that spawns endlessly).
    pids_limit: int = 128

    # Size of the writable scratch space mounted at /tmp. Matplotlib and
    # other libraries need somewhere to write temporary files even though
    # the rest of the filesystem is read-only.
    tmpfs_size: str = "64m"


# The default used everywhere unless a caller passes its own.
DEFAULT_CONFIG = SandboxConfig()

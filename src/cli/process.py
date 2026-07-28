"""Shared subprocess helpers for project commands."""

from __future__ import annotations

import shlex
import subprocess
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]


def forwarded_args(values):
    """Remove argparse's optional ``--`` forwarding separator."""
    values = list(values or [])
    return values[1:] if values[:1] == ["--"] else values


def run(command, description=None, env=None):
    """Run one project command and return its unchanged exit code."""
    command = [str(part) for part in command]
    if description:
        print(f"\n{description}", flush=True)
    print(f"  {shlex.join(command)}", flush=True)
    try:
        return subprocess.run(command, cwd=PROJECT_ROOT, env=env).returncode
    except KeyboardInterrupt:
        print("\nStopped by user.", file=sys.stderr)
        return 130


def run_module(module, args=None, description=None):
    return run(
        [sys.executable, "-m", module, *forwarded_args(args)],
        description=description,
    )


def run_script(relative_path, args=None, description=None):
    return run(
        [sys.executable, str(PROJECT_ROOT / relative_path), *forwarded_args(args)],
        description=description,
    )


def run_shell(relative_path, args=None, description=None):
    return run(
        ["bash", str(PROJECT_ROOT / relative_path), *forwarded_args(args)],
        description=description,
    )

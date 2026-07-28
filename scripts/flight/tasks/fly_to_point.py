#!/usr/bin/env python3
"""Fly one way to the currently selected destination point."""

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(PROJECT_ROOT))

from src.flight.task_runner import run_task  # noqa: E402

if __name__ == "__main__":
    raise SystemExit(run_task("fly_to_point", sys.argv[1:]))

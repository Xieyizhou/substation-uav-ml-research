#!/usr/bin/env python3
"""Preview the selected map and destination route without flying."""

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(PROJECT_ROOT))

from src.flight.task_runner import run_task  # noqa: E402

if __name__ == "__main__":
    raise SystemExit(run_task("preview_route", sys.argv[1:]))

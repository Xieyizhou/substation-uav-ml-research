"""Small subprocess wrapper that preserves an exit record across App restarts."""

from __future__ import annotations

import sys

# Direct execution otherwise places ``src/sandbox`` first on sys.path, where
# operator.py can shadow Python's standard-library operator module.
if not __package__ and sys.path:
    sys.path.pop(0)

import argparse
from datetime import datetime, timezone
import fcntl
import json
from pathlib import Path
import subprocess


def _write_result(path, exit_code, token):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps({
        "process_result_schema_version": 1,
        "exit_code": int(exit_code),
        "ended_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "ownership_token": token,
    }, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(path)


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--result", type=Path, required=True)
    parser.add_argument("--ownership", type=Path, required=True)
    parser.add_argument("--token", required=True)
    parser.add_argument("command", nargs=argparse.REMAINDER)
    args = parser.parse_args(argv)
    command = list(args.command)
    if command and command[0] == "--":
        command.pop(0)
    if not command:
        parser.error("a managed command is required")
    args.ownership.parent.mkdir(parents=True, exist_ok=True)
    ownership = args.ownership.open("w", encoding="utf-8")
    ownership.write(args.token + "\n")
    ownership.flush()
    fcntl.flock(ownership.fileno(), fcntl.LOCK_EX)
    exit_code = 1
    try:
        exit_code = subprocess.Popen(command).wait()
    except KeyboardInterrupt:
        exit_code = 130
    finally:
        _write_result(args.result, exit_code, args.token)
        fcntl.flock(ownership.fileno(), fcntl.LOCK_UN)
        ownership.close()
    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())

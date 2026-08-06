#!/bin/sh
set -eu

project_root=$(CDPATH= cd -- "$(dirname -- "$0")/.." && pwd)
cd "$project_root"

if [ ! -x .venv/bin/python ]; then
  echo "Missing .venv. Create it and install requirements.txt first." >&2
  exit 1
fi

exec .venv/bin/python main.py sandbox serve "$@"

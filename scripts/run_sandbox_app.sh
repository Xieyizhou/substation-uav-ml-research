#!/bin/sh
set -eu

project_root=$(CDPATH= cd -- "$(dirname -- "$0")/.." && pwd)
cd "$project_root"

profile=demo
if [ "${1:-}" = "--profile" ]; then
  if [ "$#" -lt 2 ]; then
    echo "--profile requires demo, development, or formal" >&2
    exit 2
  fi
  profile=$2
  shift 2
fi

python_bin=python3
if [ -x .venv/bin/python ]; then
  python_bin=.venv/bin/python
fi

"$python_bin" main.py sandbox --project-root "$project_root" \
  --profile "$profile" bootstrap >/dev/null
exec "$python_bin" main.py sandbox --project-root "$project_root" \
  --profile "$profile" serve "$@"

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

python_bin=
for candidate in .venv/bin/python "${UAV_SANDBOX_PYTHON:-}" \
  /opt/homebrew/bin/python3 /usr/local/bin/python3 /usr/bin/python3; do
  [ -n "$candidate" ] || continue
  [ -x "$candidate" ] || continue
  if "$candidate" -c 'import importlib.util,sys; raise SystemExit(0 if sys.version_info >= (3,11) and importlib.util.find_spec("mavsdk") else 1)' 2>/dev/null; then
    python_bin=$candidate
    break
  fi
done
[ -n "$python_bin" ] || {
  echo "No compatible Python 3.11+ runtime with mavsdk was found." >&2
  exit 1
}

"$python_bin" main.py sandbox --project-root "$project_root" \
  --profile "$profile" bootstrap >/dev/null
exec "$python_bin" main.py sandbox --project-root "$project_root" \
  --profile "$profile" serve "$@"

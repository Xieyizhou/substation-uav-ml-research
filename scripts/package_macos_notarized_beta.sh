#!/bin/sh
set -eu

project_root=$(CDPATH= cd -- "$(dirname -- "$0")/.." && pwd)

if [ "$#" -gt 1 ]; then
  echo "usage: $0 [version]" >&2
  exit 2
fi

exec "$project_root/scripts/package_macos_release.sh" --notarize "${1:-}"

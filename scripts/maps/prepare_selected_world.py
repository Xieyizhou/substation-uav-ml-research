#!/usr/bin/env python3
"""Create a runtime world copy with the selected red goal marker position."""

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from src.maps.goal_marker import prepare_world_with_target  # noqa: E402
from src.maps.map_catalog import map_by_id  # noqa: E402
from src.maps.target_catalog import current_target  # noqa: E402


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--map", required=True, dest="map_id")
    parser.add_argument("--source", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args(argv)

    entry = map_by_id(args.map_id)
    target = current_target(entry)
    prepare_world_with_target(args.source, args.output, entry, target)
    print(
        f"Prepared red goal marker: {target['id']} at cell "
        f"({target['cell'][0]}, {target['cell'][1]})"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

"""Record labelled RGB and truth streams for one custom-map flight."""

from __future__ import annotations

import argparse
import asyncio
import json
from pathlib import Path

from src.vision.collection.pilot_live import record_live_visual_pilot


def record_map_flight(output, context_path, flight_events, recording_id):
    context = json.loads(Path(context_path).read_text(encoding="utf-8"))
    return asyncio.run(record_live_visual_pilot(
        output,
        recording_id=recording_id,
        source_timeout_s=8.0,
        maximum_skew_ms=33.334,
        recording_context_override=context,
        flight_events_path=flight_events,
        post_landing_drain_s=1.0,
    ))


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--context", type=Path, required=True)
    parser.add_argument("--flight-events", type=Path, required=True)
    parser.add_argument("--recording-id", required=True)
    args = parser.parse_args(argv)
    record_map_flight(
        args.output, args.context, args.flight_events, args.recording_id
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

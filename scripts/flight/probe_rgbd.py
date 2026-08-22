#!/usr/bin/env python3
"""Headless Gazebo RGB-D pre-arm gate that always writes an evidence receipt."""

import argparse
import asyncio
import hashlib
import json
from pathlib import Path

from src.sensors.gazebo_rgbd import GazeboRgbDepthSource
from src.sensors.gazebo_visual_transport import inspect_visual_sources, transport_environment


def identity(payload):
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest()


async def probe(output, pair_count, timeout_s):
    receipt = {
        "schema_version": "1.0",
        "evidence_level": "blocked",
        "probe_sequence": [
            "single_topic_snapshot",
            "topic_type_inspection",
            "first_rgb",
            "first_depth",
            f"{pair_count}_synchronized_pairs",
        ],
        "transport_environment": {
            key: transport_environment()[key] for key in ("GZ_IP", "GZ_PARTITION")
        },
    }
    source = None
    try:
        inspection = await inspect_visual_sources(timeout_s=timeout_s)
        receipt["transport"] = inspection
        source = GazeboRgbDepthSource(
            output.parent / "rgbd-probe-frames",
            rgb_topic=inspection["rgb"]["topic"],
            depth_topic=inspection["depth"]["topic"],
        )
        source.discovery_receipt = inspection["topic_snapshot"]
        await source.start()
        async for _rgb, _depth, _skew_ms in source.events(timeout_s=timeout_s):
            if source.synchronizer.pair_count >= pair_count:
                break
        pairing = source.receipt()
        receipt["pairing"] = pairing
        blocked = []
        if pairing["pair_count"] < pair_count:
            blocked.append("insufficient_synchronized_pairs")
        if pairing["rgb_pairing_success_rate"] < .95:
            blocked.append("rgb_pairing_rate_below_95_percent")
        if pairing["skew_ms"]["p95"] is None or pairing["skew_ms"]["p95"] > 33.334:
            blocked.append("rgb_depth_skew_p95_exceeded")
        receipt["acceptance"] = {
            "status": "blocked" if blocked else "ready",
            "blocked_reasons": blocked,
        }
        receipt["evidence_level"] = "gazebo_live_rgbd" if not blocked else "blocked"
    except Exception as error:
        receipt["acceptance"] = {
            "status": "blocked",
            "blocked_reasons": [f"{type(error).__name__}: {error}"],
        }
    finally:
        if source is not None:
            await source.stop()
    receipt["identity"] = identity(receipt)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(receipt, indent=2, sort_keys=True) + "\n")
    return receipt


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--pairs", type=int, default=100)
    parser.add_argument("--timeout", type=float, default=20)
    args = parser.parse_args()
    receipt = asyncio.run(probe(args.output, args.pairs, args.timeout))
    print(json.dumps(receipt["acceptance"], sort_keys=True))
    return 0 if receipt["acceptance"]["status"] == "ready" else 1


if __name__ == "__main__":
    raise SystemExit(main())

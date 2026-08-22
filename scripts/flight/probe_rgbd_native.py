#!/usr/bin/env python3
"""Compile and run the native Gazebo RGB-D pre-arm pairing gate."""

import argparse
import hashlib
import json
import os
from pathlib import Path
import shlex
import subprocess

from src.sensors.gazebo_visual_transport import inspect_visual_sources

ROOT = Path(__file__).resolve().parents[2]
SOURCE = ROOT / "tools/gz_rgbd_probe.cc"
BINARY = Path("/tmp/substation-uav-gz-rgbd-probe")


def artifact_identity(payload):
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest()


def build():
    flags = subprocess.run(
        ["pkg-config", "--cflags", "--libs", "gz-transport13", "gz-msgs10"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout
    subprocess.run(
        ["clang++", str(SOURCE), "-o", str(BINARY), *shlex.split(flags)],
        check=True,
    )


async def run(args):
    environment = os.environ.copy()
    environment.setdefault("GZ_IP", "127.0.0.1")
    environment.setdefault("GZ_PARTITION", "substation_uav")
    receipt = {
        "schema_version": "1.0",
        "evidence_level": "blocked",
        "transport_environment": {
            key: environment[key] for key in ("GZ_IP", "GZ_PARTITION")
        },
    }
    try:
        receipt["transport"] = await inspect_visual_sources(timeout_s=args.timeout)
        build()
        completed = subprocess.run(
            [str(BINARY), str(args.pairs), str(args.timeout), "33.334"],
            env=environment,
            check=False,
            capture_output=True,
            text=True,
            timeout=args.timeout + 5,
        )
        if not completed.stdout.strip():
            raise RuntimeError(completed.stderr.strip() or "native probe produced no receipt")
        pairing = json.loads(completed.stdout)
        receipt["pairing"] = pairing
        blocked = []
        if completed.returncode or pairing["pair_count"] < args.pairs:
            blocked.append("insufficient_synchronized_pairs")
        if pairing["rgb_pairing_success_rate"] < .95:
            blocked.append("rgb_pairing_rate_below_95_percent")
        if pairing["skew_ms"]["p95"] < 0 or pairing["skew_ms"]["p95"] > 33.334:
            blocked.append("rgb_depth_skew_p95_exceeded")
        if pairing["rgb_size"] != [1920, 1080]:
            blocked.append("unexpected_rgb_size")
        if pairing["depth_size"] != [640, 360]:
            blocked.append("unexpected_depth_size")
        if pairing["depth_valid_ratio"] <= 0:
            blocked.append("no_valid_depth")
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
    receipt["identity"] = artifact_identity(receipt)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(receipt, indent=2, sort_keys=True) + "\n")
    return receipt


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--pairs", type=int, default=100)
    parser.add_argument("--timeout", type=float, default=20)
    args = parser.parse_args()
    import asyncio
    receipt = asyncio.run(run(args))
    print(json.dumps(receipt["acceptance"], sort_keys=True))
    return 0 if receipt["acceptance"]["status"] == "ready" else 1


if __name__ == "__main__":
    raise SystemExit(main())

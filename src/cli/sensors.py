"""Sensor diagnostics, recording, and deterministic replay commands."""

from __future__ import annotations

import argparse
import asyncio
import json
from pathlib import Path
import time

from src.sensors.factory import build_lidar_source
from src.sensors.gazebo_lidar import discover_lidar_topic
from src.sensors.replay import append_scan_record, load_scan_records


def _percentile(values, probability):
    values = sorted(values)
    if not values:
        return None
    index = min(round((len(values) - 1) * probability), len(values) - 1)
    return values[index]


def build_parser():
    parser = argparse.ArgumentParser(description="Inspect and record research sensors.")
    commands = parser.add_subparsers(dest="command", required=True)
    list_parser = commands.add_parser("list", help="Discover Gazebo sensor topics")
    list_parser.add_argument("--json", action="store_true")
    check = commands.add_parser("check", help="Check sensor readiness and health")
    check.add_argument(
        "--source", choices=["gazebo_lidar_2d", "replay"], default="gazebo_lidar_2d"
    )
    check.add_argument("--topic", default="auto")
    check.add_argument("--input", type=Path)
    check.add_argument("--timeout", type=float, default=5.0)
    record = commands.add_parser("record", help="Record Gazebo LiDAR scans as JSONL")
    record.add_argument("--topic", default="auto")
    record.add_argument("--output", type=Path, required=True)
    record.add_argument("--duration", type=float, default=10.0)
    record.add_argument(
        "--metadata",
        type=Path,
        help="Defaults to OUTPUT with a .metadata.json suffix",
    )
    replay = commands.add_parser("replay", help="Validate and summarize a scan recording")
    replay.add_argument("--input", type=Path, required=True)
    replay.add_argument("--json", action="store_true")
    return parser


async def _list_topics(as_json):
    topic = await discover_lidar_topic()
    if as_json:
        print(json.dumps({"lidar_2d": topic}))
    else:
        print(f"Gazebo 2D LiDAR: {topic}")
    return 0


async def _check(args):
    source = build_lidar_source(
        args.source,
        topic=args.topic,
        replay_path=args.input,
    )
    await source.start()
    try:
        await source.wait_ready(args.timeout)
        health = source.health()
        frame = source.latest()
        print(
            f"PASS: {health.source}, samples={len(frame.ranges_m)}, "
            f"frequency={health.frequency_hz:.1f} Hz, "
            f"age={health.last_frame_age_s:.3f}s"
        )
        return 0
    finally:
        await source.stop()


async def _record(args):
    if args.duration <= 0:
        raise ValueError("--duration must be positive")
    source = build_lidar_source("gazebo_lidar_2d", topic=args.topic)
    await source.start()
    last_sequence = None
    count = 0
    timestamps = []
    ages_ms = []
    dropped_frames = 0
    started = time.monotonic()
    try:
        await source.wait_ready(min(args.duration, 5.0))
        while time.monotonic() - started < args.duration:
            frame = source.latest()
            if frame is not None and frame.sequence != last_sequence:
                append_scan_record(args.output, frame)
                last_sequence = frame.sequence
                count += 1
                timestamps.append(frame.timestamp_s)
                health = source.health()
                dropped_frames = health.dropped_frames
                if health.last_frame_age_s is not None:
                    ages_ms.append(health.last_frame_age_s * 1000.0)
            await asyncio.sleep(0.01)
    finally:
        await source.stop()
    scan_duration = (
        max(0.0, timestamps[-1] - timestamps[0]) if len(timestamps) > 1 else 0.0
    )
    frequency_hz = (
        (len(timestamps) - 1) / scan_duration if scan_duration > 0 else 0.0
    )
    metadata = {
        "schema_version": 1,
        "record_type": "laser_scan_2d_collection",
        "path": str(args.output),
        "source": "gazebo_lidar_2d",
        "topic": source.topic,
        "frames": count,
        "duration_s": scan_duration,
        "frequency_hz": frequency_hz,
        "frame_age_p50_ms": _percentile(ages_ms, 0.50),
        "frame_age_p95_ms": _percentile(ages_ms, 0.95),
        "dropped_frames": dropped_frames,
    }
    metadata_path = args.metadata or args.output.with_suffix(".metadata.json")
    metadata_path.parent.mkdir(parents=True, exist_ok=True)
    metadata_path.write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
    print(
        f"Recorded {count} scan(s) to {args.output}; "
        f"{frequency_hz:.1f} Hz, P95 age "
        f"{metadata['frame_age_p95_ms'] or 0.0:.1f} ms"
    )
    print(f"Metadata: {metadata_path}")
    return 0 if count else 1


def _replay(args):
    frames = load_scan_records(args.input)
    duration = max(0.0, frames[-1].timestamp_s - frames[0].timestamp_s)
    frequency = (len(frames) - 1) / duration if duration > 0 else 0.0
    summary = {
        "path": str(args.input),
        "frames": len(frames),
        "duration_s": duration,
        "frequency_hz": frequency,
        "samples_per_frame": len(frames[0].ranges_m),
        "frame_id": frames[0].frame_id,
    }
    if args.json:
        print(json.dumps(summary, indent=2))
    else:
        print(
            f"PASS: {summary['frames']} frame(s), {duration:.2f}s, "
            f"{frequency:.1f} Hz, {summary['samples_per_frame']} samples/frame"
        )
    return 0


def main(argv=None):
    args = build_parser().parse_args(argv)
    try:
        if args.command == "list":
            return asyncio.run(_list_topics(args.json))
        if args.command == "check":
            return asyncio.run(_check(args))
        if args.command == "record":
            return asyncio.run(_record(args))
        if args.command == "replay":
            return _replay(args)
    except (FileNotFoundError, RuntimeError, TimeoutError, ValueError) as error:
        print(f"Sensor command failed: {error}")
        return 1
    return 2

"""Native Gazebo RGB-D source without JSON/base64 transport overhead."""

import asyncio
from dataclasses import dataclass
import json
import os
from pathlib import Path
import shlex
import subprocess
import time

ROOT = Path(__file__).resolve().parents[2]
SOURCE = ROOT / "tools/gz_rgbd_bridge.cc"
BINARY = Path("/tmp/substation-uav-gz-rgbd-bridge")


@dataclass(frozen=True)
class NativeVisualFrame:
    capture_timestamp: float
    width: int
    height: int
    payload_relative_path: str
    sequence: int
    frame_id: str


def compile_native_bridge():
    flags = subprocess.run(
        ["pkg-config", "--cflags", "--libs", "gz-transport13", "gz-msgs10"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout
    # On macOS hosts where the default SDK symlink points at a newer malformed
    # TBD archive, clang fails at link time before Gazebo is started.  Prefer a
    # known installed SDK as a toolchain repair only; no image or timing logic
    # changes.  An explicit caller SDKROOT remains authoritative.
    environment = os.environ.copy()
    if not environment.get("SDKROOT"):
        for candidate in (
            "/Library/Developer/CommandLineTools/SDKs/MacOSX26.5.sdk",
            "/Library/Developer/CommandLineTools/SDKs/MacOSX26.sdk",
            "/Library/Developer/CommandLineTools/SDKs/MacOSX.sdk",
        ):
            if Path(candidate).exists():
                environment["SDKROOT"] = candidate
                break
    subprocess.run(
        ["clang++", str(SOURCE), "-o", str(BINARY), *shlex.split(flags)],
        check=True,
        env=environment,
    )


class NativeGazeboRgbDepthSource:
    def __init__(self, staging_directory, *, emit_stride=6, maximum_skew_ms=33.334, preserve_frames=False):
        self.staging = Path(staging_directory)
        self.emit_stride = int(emit_stride)
        self.maximum_skew_ms = float(maximum_skew_ms)
        self.preserve_frames = bool(preserve_frames)
        self.process = None
        self.latest = None
        self.started_monotonic = None

    async def start(self):
        self.staging.mkdir(parents=True, exist_ok=True)
        await asyncio.to_thread(compile_native_bridge)
        environment = os.environ.copy()
        environment.setdefault("GZ_IP", "127.0.0.1")
        environment.setdefault("GZ_PARTITION", "substation_uav")
        self.process = await asyncio.create_subprocess_exec(
            str(BINARY),
            str(self.staging),
            str(self.emit_stride),
            str(self.maximum_skew_ms),
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
            env=environment,
        )
        self.started_monotonic = time.monotonic()

    async def events(self, *, timeout_s):
        if self.process is None or self.process.stdout is None:
            raise RuntimeError("native RGB-D source is not started")
        previous = None
        try:
            while True:
                line = await asyncio.wait_for(
                    self.process.stdout.readline(), timeout=timeout_s
                )
                if not line:
                    error = ""
                    if self.process.stderr is not None:
                        error = (await self.process.stderr.read()).decode(
                            errors="replace"
                        )
                    raise RuntimeError(
                        f"native RGB-D bridge stopped: {error.strip()}"
                    )
                row = json.loads(line)
                self.latest = row
                if previous is not None and not self.preserve_frames:
                    for relative_path in previous:
                        (self.staging / relative_path).unlink(missing_ok=True)
                previous = (row["rgb_path"], row["depth_path"])
                rgb = NativeVisualFrame(
                    capture_timestamp=row["rgb_timestamp"],
                    width=row["rgb_width"],
                    height=row["rgb_height"],
                    payload_relative_path=row["rgb_path"],
                    sequence=row["pair_count"],
                    frame_id="research_rgb",
                )
                depth = NativeVisualFrame(
                    capture_timestamp=row["depth_timestamp"],
                    width=row["depth_width"],
                    height=row["depth_height"],
                    payload_relative_path=row["depth_path"],
                    sequence=row["pair_count"],
                    frame_id="research_depth",
                )
                yield rgb, depth, row["skew_ms"]
        finally:
            if previous is not None and not self.preserve_frames:
                for relative_path in previous:
                    (self.staging / relative_path).unlink(missing_ok=True)

    async def stop(self):
        if self.process is None:
            return
        if self.process.returncode is None:
            self.process.terminate()
        try:
            await asyncio.wait_for(self.process.communicate(), timeout=3)
        except asyncio.TimeoutError:
            self.process.kill()
            await self.process.communicate()

    def receipt(self):
        row = self.latest or {}
        return {
            "transport": "gz-transport13-native",
            "rgb_frame_count": row.get("rgb_frame_count", 0),
            "depth_frame_count": row.get("depth_frame_count", 0),
            "pair_count": row.get("pair_count", 0),
            "dropped_rgb_count": row.get("dropped_rgb_count", 0),
            "dropped_depth_count": row.get("dropped_depth_count", 0),
            "stale_depth_count": row.get("dropped_depth_count", 0),
            "pairing_success_rate": row.get("rgb_pairing_success_rate", 0),
            "rgb_pairing_success_rate": row.get("rgb_pairing_success_rate", 0),
            "depth_pairing_success_rate": row.get("depth_pairing_success_rate", 0),
            "skew_ms": {
                "p50": row.get("skew_p50_ms"),
                "p95": row.get("skew_p95_ms"),
                "max": row.get("skew_max_ms"),
            },
            "maximum_skew_ms": self.maximum_skew_ms,
            "emit_stride": self.emit_stride,
            "preserve_frames": self.preserve_frames,
        }

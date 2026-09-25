"""Bounded in-memory RGB/depth/estimator pairing for Gazebo PX4 SITL."""

import asyncio
from collections import Counter, deque
from pathlib import Path
import time

from src.flight.sim_pose_sync import MavsdkSimPoseSource
from src.sensors.gazebo_camera_latest import GazeboLatestMemorySource
from src.sensors.gazebo_depth_memory import aligned_depth, decode_depth


class AlignedSimRgbdSource:
    def __init__(self, drone, *, gazebo_sitl=False):
        self.pose = MavsdkSimPoseSource(drone, gazebo_sitl=gazebo_sitl)
        self.rgb = GazeboLatestMemorySource(Path("unused-memory"), topic="/research_camera/image")
        self.depth = GazeboLatestMemorySource(Path("unused-memory"), topic="/research_camera/depth")
        self.rgb_queue, self.depth_queue = asyncio.Queue(1), asyncio.Queue(1)
        self.depth_frames, self.tasks, self.counts = deque(maxlen=8), [], Counter()

    async def start(self):
        try:
            await self.pose.start()
            await self.rgb.start()
            await self.depth.start()
            self.tasks = [asyncio.create_task(self._receive(self.rgb, self.rgb_queue, "rgb")),
                          asyncio.create_task(self._receive(self.depth, self.depth_queue, "depth")),
                          asyncio.create_task(self._decode_depth())]
        except BaseException:
            await self.stop()
            raise

    async def _receive(self, source, queue, name):
        async for raw in source.events(timeout_s=5):
            self.counts[name+"_received"] += 1
            if queue.full():
                queue.get_nowait()
                self.counts[name+"_replaced"] += 1
            queue.put_nowait(raw)
        raise RuntimeError(name + " stream ended")

    async def _decode_depth(self):
        while True:
            raw = await self.depth_queue.get()
            if not 0 <= time.monotonic()-raw.received <= .5:
                self.counts["depth_stale_before_decode"] += 1
                continue
            try:
                depth = await asyncio.to_thread(decode_depth, raw)
            except (ValueError, TypeError):
                self.counts["depth_invalid"] += 1
                continue
            if self.depth_frames and depth.capture_timestamp <= self.depth_frames[-1].capture_timestamp:
                raise RuntimeError("depth capture clock stopped or reversed")
            self.depth_frames.append(depth)

    def check_health(self):
        for task in [*self.tasks, *self.pose.tasks]:
            if task.done():
                task.result()
                raise RuntimeError("RGB-D/pose task ended")

    async def events(self):
        previous = None
        while True:
            self.check_health()
            try:
                raw = await asyncio.wait_for(self.rgb_queue.get(), 1)
            except asyncio.TimeoutError:
                self.check_health()
                raise TimeoutError("aligned RGB stream unavailable")
            if not 0 <= time.monotonic()-raw.received <= .5:
                self.counts["rgb_stale_before_decode"] += 1
                continue
            try:
                event = await asyncio.to_thread(raw.materialize)
            except (ValueError, TypeError):
                self.counts["rgb_invalid"] += 1
                continue
            if previous is not None and event.frame.capture_timestamp <= previous:
                raise RuntimeError("RGB capture clock stopped or reversed")
            previous = event.frame.capture_timestamp
            # Allow bounded arrival skew; never extrapolate estimator poses.
            deadline, reason = time.monotonic()+.1, "waiting_for_depth"
            matched = None
            while time.monotonic() < deadline:
                self.check_health()
                if self.depth_frames:
                    depth = min(self.depth_frames, key=lambda d: abs(d.capture_timestamp-previous))
                    try:
                        now = time.monotonic()
                        aligned_depth(event.frame, depth, now=now)
                        matched = self.pose.match(event.frame, now=now)
                        break
                    except ValueError as exc:
                        reason = str(exc)
                await asyncio.sleep(.005)
            if matched is None:
                self.counts["rejected: "+reason] += 1
                continue
            self.counts["paired"] += 1
            yield event, depth, matched

    async def stop(self):
        for task in self.tasks:
            task.cancel()
        await asyncio.gather(*self.tasks, return_exceptions=True)
        self.tasks = []
        results = await asyncio.gather(self.rgb.stop(), self.depth.stop(), self.pose.stop(), return_exceptions=True)
        for result in results:
            if isinstance(result, BaseException):
                raise result

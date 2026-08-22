"""Paired Gazebo RGB and depth source."""

import asyncio

from src.sensors.gazebo_camera import GazeboCameraSource
from src.sensors.gazebo_depth import GazeboDepthSource
from src.sensors.gazebo_visual_transport import (
    DEPTH_CONFIGURED_TOPIC,
    RGB_CONFIGURED_TOPIC,
    discover_configured_topics,
)
from src.sensors.rgb_depth_sync import RgbDepthSynchronizer


class GazeboRgbDepthSource:
    def __init__(self, staging_directory, *, rgb_topic="auto", depth_topic="auto", maximum_skew_ms=33.334):
        self.rgb = GazeboCameraSource(staging_directory, topic=rgb_topic)
        self.depth = GazeboDepthSource(staging_directory, topic=depth_topic)
        self.synchronizer = RgbDepthSynchronizer(maximum_skew_ms)
        self.discovery_receipt = None

    async def start(self):
        if self.rgb.topic == "auto" or self.depth.topic == "auto":
            configured = {
                "rgb": RGB_CONFIGURED_TOPIC if self.rgb.topic == "auto" else self.rgb.topic,
                "depth": DEPTH_CONFIGURED_TOPIC if self.depth.topic == "auto" else self.depth.topic,
            }
            self.discovery_receipt = await discover_configured_topics(configured)
            resolved = self.discovery_receipt["resolved_topics"]
            self.rgb.topic = resolved["rgb"]
            self.depth.topic = resolved["depth"]
        await asyncio.gather(self.rgb.start(), self.depth.start())

    async def events(self, *, timeout_s):
        rgb_events = self.rgb.events(timeout_s=timeout_s)
        depth_events = self.depth.events(timeout_s=timeout_s)
        tasks = {
            asyncio.create_task(anext(rgb_events)): "rgb",
            asyncio.create_task(anext(depth_events)): "depth",
        }
        try:
            while tasks:
                done, _ = await asyncio.wait(tasks, return_when=asyncio.FIRST_COMPLETED)
                for task in done:
                    source = tasks.pop(task)
                    value = task.result()
                    if source == "rgb":
                        if value.valid:
                            pairs = self.synchronizer.push_rgb(value.frame)
                        tasks[asyncio.create_task(anext(rgb_events))] = "rgb"
                    else:
                        pairs = self.synchronizer.push_depth(value)
                        tasks[asyncio.create_task(anext(depth_events))] = "depth"
                    for pair in pairs:
                        yield pair
        finally:
            for task in tasks:
                task.cancel()
            await asyncio.gather(*tasks, return_exceptions=True)
            await rgb_events.aclose()
            await depth_events.aclose()

    async def stop(self):
        await asyncio.gather(self.rgb.stop(), self.depth.stop())

    def receipt(self):
        return {
            **self.synchronizer.receipt(),
            "topic_discovery": self.discovery_receipt,
            "rgb_health": self.rgb.health().__dict__,
            "depth_health": self.depth.health().__dict__,
        }

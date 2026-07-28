"""Construct sensor sources without leaking simulator details into flight code."""

from pathlib import Path

from src.sensors.gazebo_lidar import GazeboLidar2DSource
from src.sensors.replay import ReplayLidarSource


def build_lidar_source(source_id, *, topic="auto", replay_path=None, stale_after_s=0.5):
    if source_id == "gazebo_lidar_2d":
        return GazeboLidar2DSource(topic=topic, stale_after_s=stale_after_s)
    if source_id == "replay":
        if replay_path is None:
            raise ValueError("--sensor-replay is required for replay perception")
        return ReplayLidarSource(
            Path(replay_path),
            loop=True,
            stale_after_s=stale_after_s,
        )
    raise ValueError(f"unsupported LiDAR source: {source_id}")

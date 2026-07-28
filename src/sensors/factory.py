"""Construct sensor sources without leaking simulator details into flight code."""

from pathlib import Path

from src.sensors.gazebo_lidar import GazeboLidar2DSource
from src.sensors.replay import ReplayLidarSource


def build_lidar_source(
    source_id,
    *,
    topic="auto",
    replay_path=None,
    stale_after_s=0.5,
    scenario_manifest=None,
):
    if source_id == "gazebo_lidar_2d":
        source = GazeboLidar2DSource(topic=topic, stale_after_s=stale_after_s)
    elif source_id == "replay":
        if replay_path is None:
            raise ValueError("--sensor-replay is required for replay perception")
        source = ReplayLidarSource(
            Path(replay_path),
            loop=True,
            stale_after_s=stale_after_s,
        )
    else:
        raise ValueError(f"unsupported LiDAR source: {source_id}")
    if scenario_manifest:
        from src.sensors.fault_injection import FaultInjectedLidarSource

        return FaultInjectedLidarSource.from_path(source, scenario_manifest)
    return source

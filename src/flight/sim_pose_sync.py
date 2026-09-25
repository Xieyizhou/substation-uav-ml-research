"""Timestamped estimator poses for PX4 Gazebo SITL (not physical vehicles).

PX4 GZBridge synchronizes HRT to Gazebo simulation time. ATTITUDE and
LOCAL_POSITION_NED contain publication time_boot_ms, not host receive time.
The explicit SITL contract and bounded skew are required; no clock offset is
guessed from message arrival. These timestamps are not estimator sample times.
"""

import asyncio
from collections import deque
from dataclasses import dataclass
import json
import math
import time

from src.sensors.gazebo_visual_transport import GAZEBO_SIM_CLOCK


@dataclass(frozen=True)
class PoseSample:
    timestamp: float
    received: float
    values: dict


class SimPoseBuffer:
    def __init__(self, *, capacity=120, maximum_skew_s=.05, maximum_receive_age_s=.5,
                 maximum_capture_age_s=.2):
        if type(capacity) is not int or not 2 <= capacity <= 1000:
            raise ValueError("pose buffer capacity must be in [2,1000]")
        for value in (maximum_skew_s, maximum_receive_age_s, maximum_capture_age_s):
            if not math.isfinite(value) or value <= 0:
                raise ValueError("pose timing limits must be finite and positive")
        self.position, self.attitude = deque(maxlen=capacity), deque(maxlen=capacity)
        self.maximum_skew_s = maximum_skew_s
        self.maximum_receive_age_s = maximum_receive_age_s
        self.maximum_capture_age_s = maximum_capture_age_s
        self.source = None
        self.fault = None

    def add(self, name, fields, *, received, system_id, component_id):
        if self.fault:
            raise ValueError(self.fault)
        source = (system_id, component_id)
        if self.source is not None and source != self.source:
            self.fault = "estimator source changed"
            raise ValueError(self.fault)
        self.source = source
        stamp = float(fields["time_boot_ms"]) / 1000
        if name == "LOCAL_POSITION_NED":
            values = dict(north_m=float(fields["x"]), east_m=float(fields["y"]),
                          altitude_m=-float(fields["z"]), vn=float(fields["vx"]),
                          ve=float(fields["vy"]), vd=float(fields["vz"]))
            queue = self.position
        elif name == "ATTITUDE":
            values = {key + "_deg": math.degrees(float(fields[key])) for key in ("roll", "pitch", "yaw")}
            queue = self.attitude
        else:
            raise ValueError("unsupported pose message")
        if not all(math.isfinite(v) for v in (stamp, received, *values.values())) or stamp < 0 or received < 0:
            self.fault = "nonfinite or negative pose timestamp/value"
            raise ValueError(self.fault)
        if queue and stamp < queue[-1].timestamp:
            self.fault = "pose clock reversed or reset"
            raise ValueError(self.fault)
        if queue and stamp == queue[-1].timestamp:
            return  # A duplicate cannot refresh an old estimator pose.
        queue.append(PoseSample(stamp, received, values))

    def match(self, capture_timestamp, clock_domain, *, now):
        if self.fault:
            raise ValueError(self.fault)
        if clock_domain != GAZEBO_SIM_CLOCK:
            raise ValueError("image clock is not Gazebo simulation time")
        if not all(math.isfinite(v) and v >= 0 for v in (capture_timestamp, now)):
            raise ValueError("invalid capture or receive time")
        if not self.position or not self.attitude:
            raise ValueError("waiting for timestamped estimator pose")
        selected = []
        for queue in (self.position, self.attitude):
            if not 0 <= now - queue[-1].received <= self.maximum_receive_age_s:
                raise ValueError("estimator stream stale")
            if not -self.maximum_skew_s <= queue[-1].timestamp - capture_timestamp <= self.maximum_capture_age_s:
                raise ValueError("image capture time stale or clocks disagree")
            sample = min(queue, key=lambda s: abs(s.timestamp - capture_timestamp))
            if abs(sample.timestamp - capture_timestamp) > self.maximum_skew_s:
                raise ValueError("no estimator pose within capture skew")
            if not 0 <= now - sample.received <= self.maximum_receive_age_s:
                raise ValueError("matched estimator pose stale")
            selected.append(sample)
        position, attitude = selected
        return {"pose": {**position.values, **attitude.values},
                "capture_timestamp": capture_timestamp, "clock_domain": clock_domain,
                "position_timestamp": position.timestamp, "attitude_timestamp": attitude.timestamp,
                "position_skew_s": position.timestamp-capture_timestamp,
                "attitude_skew_s": attitude.timestamp-capture_timestamp,
                "estimator_system_id": self.source[0], "estimator_component_id": self.source[1],
                "timestamp_scope": "PX4 estimator publication, not sensor sample time"}


class MavsdkSimPoseSource:
    def __init__(self, drone, *, gazebo_sitl=False):
        if gazebo_sitl is not True:
            raise ValueError("timestamp matching requires an explicit Gazebo SITL contract")
        self.drone, self.buffer, self.tasks = drone, SimPoseBuffer(), []

    async def start(self):
        if self.tasks:
            raise RuntimeError("pose source already started")
        for name in ("LOCAL_POSITION_NED", "ATTITUDE"):
            self.tasks.append(asyncio.create_task(self._read(name), name="pose-" + name))
        try:
            await asyncio.wait_for(self.drone.telemetry.set_rate_position_velocity_ned(30), 5)
            await asyncio.wait_for(self.drone.telemetry.set_rate_attitude_euler(30), 5)
        except BaseException:
            await self.stop()
            raise

    async def _read(self, name):
        async for message in self.drone.mavlink_direct.message(name):
            self.buffer.add(name, json.loads(message.fields_json), received=time.monotonic(),
                            system_id=message.system_id, component_id=message.component_id)
        raise RuntimeError("timestamped pose stream ended: " + name)

    def match(self, frame, *, now=None):
        if not self.tasks:
            raise RuntimeError("pose source is not running")
        for task in self.tasks:
            if task.done():
                task.result()
                raise RuntimeError("pose subscription ended")
        return self.buffer.match(frame.capture_timestamp, frame.capture_clock_domain,
                                 now=time.monotonic() if now is None else now)

    async def stop(self):
        for task in self.tasks:
            task.cancel()
        await asyncio.gather(*self.tasks, return_exceptions=True)
        self.tasks = []

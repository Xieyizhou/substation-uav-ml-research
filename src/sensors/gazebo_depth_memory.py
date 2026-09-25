"""Decode only selected depth messages; retain no per-frame disk payloads."""

from dataclasses import dataclass
import hashlib
import json
import math

import numpy as np

from src.sensors.gazebo_depth_codec import FLOAT32_FORMATS
from src.sensors.gazebo_image_codec import _message_bytes
from src.sensors.gazebo_visual_transport import GAZEBO_SIM_CLOCK, message_timestamp


@dataclass(frozen=True)
class MemoryDepth:
    capture_timestamp: float
    receive_monotonic_timestamp: float
    array: object
    payload_sha256: str
    capture_clock_domain: str = GAZEBO_SIM_CLOCK


def decode_depth(raw, *, max_pixels=640*360):
    message = json.loads(raw.line)
    width, height = int(message.get("width", 0)), int(message.get("height", 0))
    if min(width, height) <= 0 or width * height > max_pixels:
        raise ValueError("depth dimensions exceed bounded input")
    if message.get("pixel_format_type", message.get("pixelFormatType")) not in FLOAT32_FORMATS:
        raise ValueError("unsupported depth format")
    payload = _message_bytes(message.get("data"))
    stride = int(message.get("step", width*4))
    if stride < width*4 or stride % 4 or len(payload) != stride*height:
        raise ValueError("invalid depth payload layout")
    array = np.frombuffer(payload, dtype="<f4").reshape(height, stride//4)[:, :width].copy()
    array.setflags(write=False)
    timestamp = message_timestamp(message)
    if not math.isfinite(timestamp) or not math.isfinite(raw.received):
        raise ValueError("nonfinite depth timing")
    return MemoryDepth(timestamp, raw.received, array, hashlib.sha256(array.tobytes()).hexdigest())


def aligned_depth(rgb, depth, *, now, maximum_skew_s=.033334, maximum_receive_age_s=.5):
    if not all(math.isfinite(value) and value >= 0 for value in
               (rgb.capture_timestamp, depth.capture_timestamp, now)):
        raise ValueError("invalid RGB/depth capture time")
    if rgb.capture_clock_domain != depth.capture_clock_domain or depth.capture_clock_domain != GAZEBO_SIM_CLOCK:
        raise ValueError("RGB/depth clock mismatch")
    if abs(rgb.capture_timestamp-depth.capture_timestamp) > maximum_skew_s:
        raise ValueError("RGB/depth capture skew exceeded")
    ages = [now-frame.receive_monotonic_timestamp for frame in (rgb, depth)]
    if not all(math.isfinite(age) and 0 <= age <= maximum_receive_age_s for age in ages):
        raise ValueError("RGB/depth receive age exceeded")
    return depth


def bbox_depth(depth, rgb_size, bbox, *, minimum_pixels=16, maximum_spread_m=.5):
    width, height = rgb_size
    if min(width, height) <= 0 or len(bbox) != 4 or not all(math.isfinite(v) for v in bbox):
        return None
    x1, y1, x2, y2 = bbox
    if not (0 < x1 < x2 < width and 0 < y1 < y2 < height):
        return None
    dw, dh = depth.array.shape[1], depth.array.shape[0]
    cx, cy, hw, hh = (x1+x2)/2, (y1+y2)/2, (x2-x1)*.2, (y2-y1)*.2
    patch = depth.array[math.floor((cy-hh)*dh/height):math.ceil((cy+hh)*dh/height),
                        math.floor((cx-hw)*dw/width):math.ceil((cx+hw)*dw/width)]
    values = patch[np.isfinite(patch) & (patch >= .2) & (patch <= 100)]
    if len(values) < minimum_pixels or len(values) < patch.size*.8:
        return None
    if float(np.quantile(values, .9)-np.quantile(values, .1)) > maximum_spread_m:
        return None
    return float(np.median(values))

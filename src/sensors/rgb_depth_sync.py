"""Deterministic Gazebo simulation-time RGB/depth pairing and robust range extraction."""

from collections import deque
import math
from pathlib import Path
import statistics
import struct


class RgbDepthSynchronizer:
    def __init__(self, maximum_skew_ms=33.334, capacity=8):
        self.maximum_skew_s = float(maximum_skew_ms) / 1000
        self.rgb, self.depth = deque(maxlen=capacity), deque(maxlen=capacity)
        self.rgb_count = self.depth_count = self.pair_count = 0
        self.dropped_rgb_count = self.dropped_depth_count = 0
        self.stale_depth_count = 0
        self.skews_ms = []

    def push_rgb(self, frame):
        self.rgb_count += 1
        self.rgb.append(frame)
        return self._pairs()

    def push_depth(self, frame):
        self.depth_count += 1
        self.depth.append(frame)
        return self._pairs()

    def _pairs(self):
        pairs = []
        while self.rgb and self.depth:
            rgb = self.rgb[0]
            depth = min(self.depth, key=lambda item: abs(item.capture_timestamp - rgb.capture_timestamp))
            skew = abs(depth.capture_timestamp - rgb.capture_timestamp)
            if skew <= self.maximum_skew_s:
                self.rgb.popleft(); self.depth.remove(depth); self.pair_count += 1
                self.skews_ms.append(skew * 1000)
                pairs.append((rgb, depth, skew * 1000))
            elif depth.capture_timestamp > rgb.capture_timestamp:
                self.rgb.popleft(); self.dropped_rgb_count += 1
            else:
                self.depth.remove(depth)
                self.dropped_depth_count += 1
                self.stale_depth_count += 1
        return pairs

    def receipt(self):
        skews = sorted(self.skews_ms)
        percentile = lambda q: (
            skews[max(0, math.ceil(q * len(skews)) - 1)] if skews else None
        )
        rgb_rate = self.pair_count / self.rgb_count if self.rgb_count else 0
        depth_rate = self.pair_count / self.depth_count if self.depth_count else 0
        return {
            "rgb_count": self.rgb_count,
            "rgb_frame_count": self.rgb_count,
            "depth_frame_count": self.depth_count,
            "pair_count": self.pair_count,
            "dropped_rgb_count": self.dropped_rgb_count,
            "dropped_depth_count": self.dropped_depth_count,
            "stale_depth_count": self.stale_depth_count,
            "pairing_success_rate": rgb_rate,
            "rgb_pairing_success_rate": rgb_rate,
            "depth_pairing_success_rate": depth_rate,
            "skew_ms": {
                "p50": percentile(.5),
                "p95": percentile(.95),
                "max": max(skews) if skews else None,
            },
            "maximum_skew_ms": self.maximum_skew_s * 1000,
        }


def depth_for_bbox(depth_frame, rgb_size, bbox, staging_directory, *, inner_fraction=.4, minimum_pixels=16):
    raw = (Path(staging_directory) / depth_frame.payload_relative_path).read_bytes()
    values = struct.unpack(f"<{depth_frame.width * depth_frame.height}f", raw)
    rgb_w, rgb_h = rgb_size
    x1, y1, x2, y2 = [float(value) for value in bbox]
    center_x, center_y = (x1 + x2) / 2, (y1 + y2) / 2
    half_w, half_h = (x2 - x1) * inner_fraction / 2, (y2 - y1) * inner_fraction / 2
    left = max(0, math.floor((center_x - half_w) * depth_frame.width / rgb_w))
    right = min(depth_frame.width, math.ceil((center_x + half_w) * depth_frame.width / rgb_w))
    top = max(0, math.floor((center_y - half_h) * depth_frame.height / rgb_h))
    bottom = min(depth_frame.height, math.ceil((center_y + half_h) * depth_frame.height / rgb_h))
    valid = [values[y * depth_frame.width + x] for y in range(top, bottom) for x in range(left, right)
             if math.isfinite(values[y * depth_frame.width + x]) and .2 <= values[y * depth_frame.width + x] <= 100]
    return statistics.median(valid) if len(valid) >= minimum_pixels else None

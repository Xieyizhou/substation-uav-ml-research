"""Associate camera equipment detections with LiDAR range evidence."""

from __future__ import annotations

from dataclasses import replace
import math


def associate_equipment_with_lidar(
    detections,
    scan,
    *,
    image_width,
    camera_hfov_deg,
    vehicle_north_m,
    vehicle_east_m,
    vehicle_down_m,
    yaw_deg,
    angular_window_deg=2.0,
):
    if image_width <= 0 or camera_hfov_deg <= 0:
        raise ValueError("camera width and FOV must be positive")
    associated = []
    for detection in detections:
        x_min, _, x_max, _ = detection.bbox_xyxy
        center_x = (x_min + x_max) / 2.0
        relative_camera_deg = (0.5 - center_x / image_width) * camera_hfov_deg
        target_scan_angle = math.radians(relative_camera_deg)
        ranges = []
        for index, distance in enumerate(scan.ranges_m):
            if (
                math.isfinite(distance)
                and scan.range_min_m <= distance <= scan.range_max_m
                and abs(scan.angle_at(index) - target_scan_angle)
                <= math.radians(angular_window_deg)
            ):
                ranges.append(distance)
        if not ranges:
            associated.append(detection)
            continue
        distance = sorted(ranges)[len(ranges) // 2]
        compass_bearing = math.radians(yaw_deg - relative_camera_deg)
        position = (
            vehicle_north_m + distance * math.cos(compass_bearing),
            vehicle_east_m + distance * math.sin(compass_bearing),
            vehicle_down_m,
        )
        associated.append(replace(detection, position_ned_m=position))
    return tuple(associated)

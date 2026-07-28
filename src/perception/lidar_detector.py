"""Geometric LiDAR risk and traversability baseline."""

from __future__ import annotations

import math
import time

from src.perception.local_costmap import RollingCostmapBuilder
from src.sensors.types import RiskEstimate


class LidarRiskDetector:
    """Adapt an async scan source to the project's synchronous detector contract."""

    def __init__(
        self,
        source,
        *,
        resolution_m,
        detection_range_m=10.0,
        warning_distance_m=2.0,
        danger_distance_m=1.0,
        detection_fov_deg=90.0,
        stale_after_s=0.5,
        reaction_time_s=0.5,
        braking_deceleration_m_s2=1.5,
        nominal_speed_m_s=1.0,
        inflation_radius_m=0.5,
        risk_predictor=None,
        risk_fusion="safety_max",
    ):
        self.source = source
        self.resolution_m = float(resolution_m)
        self.detection_range_m = float(detection_range_m)
        self.warning_distance_m = float(warning_distance_m)
        self.danger_distance_m = float(danger_distance_m)
        self.detection_fov_deg = float(detection_fov_deg)
        self.stale_after_s = float(stale_after_s)
        self.reaction_time_s = float(reaction_time_s)
        self.braking_deceleration_m_s2 = float(braking_deceleration_m_s2)
        self.nominal_speed_m_s = float(nominal_speed_m_s)
        self.inflation_radius_m = float(inflation_radius_m)
        self.risk_predictor = risk_predictor
        if risk_fusion not in {"ml_only", "safety_max"}:
            raise ValueError("risk_fusion must be ml_only or safety_max")
        self.risk_fusion = risk_fusion
        self.last_costmap = None
        self.last_risk = None
        self.costmap_builder = RollingCostmapBuilder(
            resolution_m=0.25,
            forward_range_m=self.detection_range_m,
            lateral_range_m=self.detection_range_m * 2.0,
            inflation_radius_m=self.inflation_radius_m,
            decay_half_life_s=1.0,
        )

    async def start(self):
        await self.source.start()

    async def wait_ready(self, timeout_s):
        await self.source.wait_ready(timeout_s)

    async def stop(self):
        await self.source.stop()

    def _stopping_distance(self):
        braking = self.nominal_speed_m_s**2 / (
            2.0 * max(self.braking_deceleration_m_s2, 0.01)
        )
        return self.nominal_speed_m_s * self.reaction_time_s + braking

    def _global_cell(self, distance_m, scan_angle_rad, north_m, east_m, yaw_deg):
        # Gazebo scan angles are positive to body-left. Local NED compass yaw
        # is positive to the east, so relative compass bearing is -scan angle.
        bearing_rad = math.radians(yaw_deg or 0.0) - scan_angle_rad
        obstacle_north = north_m + distance_m * math.cos(bearing_rad)
        obstacle_east = east_m + distance_m * math.sin(bearing_rad)
        return (
            int(math.floor(obstacle_east / self.resolution_m)),
            int(math.floor(obstacle_north / self.resolution_m)),
            obstacle_north,
            obstacle_east,
        )

    def detect(self, local_north_m, local_east_m, yaw_deg=None, altitude_m=None):
        started = time.perf_counter()
        scan = self.source.latest()
        health = self.source.health()
        if scan is None or not health.healthy or (
            health.last_frame_age_s is not None
            and health.last_frame_age_s > self.stale_after_s
        ):
            reason = health.message or "LiDAR frame unavailable"
            return self._unhealthy_detection(reason, health, started)

        self.last_costmap = self.costmap_builder.update(scan)
        detected = []
        half_fov_rad = math.radians(self.detection_fov_deg) / 2.0
        for index, distance_m in enumerate(scan.ranges_m):
            if (
                not math.isfinite(distance_m)
                or distance_m < scan.range_min_m
                or distance_m > min(scan.range_max_m, self.detection_range_m)
            ):
                continue
            angle_rad = scan.angle_at(index)
            if abs(angle_rad) > half_fov_rad:
                continue
            grid_x, grid_y, north_m, east_m = self._global_cell(
                distance_m,
                angle_rad,
                float(local_north_m),
                float(local_east_m),
                float(yaw_deg or 0.0),
            )
            detected.append(
                {
                    "obstacle_layer": "lidar",
                    "grid_x": grid_x,
                    "grid_y": grid_y,
                    "obstacle_east_m": east_m,
                    "obstacle_north_m": north_m,
                    "obstacle_name": "lidar_return",
                    "obstacle_type": "unknown",
                    "distance_m": distance_m,
                    "bearing_deg_relative": -math.degrees(angle_rad),
                    "in_detection_range": True,
                    "in_fov": True,
                    "detected": True,
                }
            )
        detected.sort(key=lambda item: item["distance_m"])
        nearest = detected[0] if detected else None
        nearest_distance = nearest["distance_m"] if nearest else None
        stopping_distance = self._stopping_distance()
        if nearest_distance is None:
            level = "clear"
            reason = "no valid LiDAR return in the forward corridor"
        elif nearest_distance <= max(self.danger_distance_m, stopping_distance):
            level = "danger"
            reason = "obstacle is inside stopping distance"
        elif nearest_distance <= self.warning_distance_m:
            level = "warning"
            reason = "obstacle is inside warning distance"
        else:
            level = "detected"
            reason = "obstacle is inside LiDAR detection corridor"
        collision_time = (
            nearest_distance / max(self.nominal_speed_m_s, 0.01)
            if nearest_distance is not None
            else None
        )
        model_id = "geometric_lidar_v1"
        confidence = 1.0 if nearest else 0.95
        recommended_direction = self._recommended_direction(scan)
        model_latency_ms = 0.0
        if self.risk_predictor is not None:
            prediction = self.risk_predictor.predict(scan)
            order = {"clear": 0, "detected": 1, "warning": 2, "danger": 3}
            predicted_level = prediction["risk_level"]
            if self.risk_fusion == "ml_only":
                level = predicted_level
                reason = "ML risk selected for candidate comparison"
                model_id = prediction["model_id"]
            elif order.get(predicted_level, 0) > order.get(level, 0):
                level = predicted_level
                reason = "ML risk exceeded geometric safety risk"
            confidence = float(prediction["confidence"])
            recommended_direction = prediction["recommended_direction_deg"]
            model_latency_ms = float(prediction["latency_ms"])
            if self.risk_fusion == "safety_max":
                model_id = f"hybrid:{prediction['model_id']}+geometric_lidar_v1"
        latency_ms = (time.perf_counter() - started) * 1000.0
        self.last_risk = RiskEstimate(
            level=level,
            confidence=confidence,
            nearest_distance_m=nearest_distance,
            collision_time_s=collision_time,
            recommended_direction_deg=recommended_direction,
            inference_latency_ms=latency_ms,
            model_id=model_id,
            reason=reason,
        )
        return {
            "detected_obstacle": bool(detected),
            "detected": bool(detected),
            "risk_level": level,
            "closest_obstacle_name": nearest["obstacle_name"] if nearest else "",
            "closest_obstacle_distance_m": nearest_distance,
            "closest_obstacle_bearing_deg": (
                nearest["bearing_deg_relative"] if nearest else None
            ),
            "closest_obstacle_in_detection_range": bool(nearest),
            "closest_obstacle_in_fov": bool(nearest),
            "nearest_obstacle_name": nearest["obstacle_name"] if nearest else "",
            "nearest_obstacle_layer": "lidar" if nearest else "",
            "nearest_obstacle_distance_m": nearest_distance,
            "nearest_obstacle_bearing_deg": (
                nearest["bearing_deg_relative"] if nearest else None
            ),
            "detected_obstacle_count": len(detected),
            "warning_distance_m": self.warning_distance_m,
            "danger_distance_m": self.danger_distance_m,
            "closest_obstacle": nearest,
            "nearest_obstacle": nearest,
            "detected_obstacles": detected,
            "dynamic_grid_cells": self._global_cells_from_costmap(
                self.last_costmap,
                float(local_north_m),
                float(local_east_m),
                float(yaw_deg or 0.0),
            ),
            "sensor_source": self.source.source_id,
            "sensor_healthy": True,
            "sensor_message": "",
            "sensor_frequency_hz": health.frequency_hz,
            "sensor_frame_age_s": health.last_frame_age_s,
            "sensor_dropped_frames": health.dropped_frames,
            "costmap": self.last_costmap,
            "risk_estimate": self.last_risk,
            "risk_model_id": self.last_risk.model_id,
            "inference_latency_ms": latency_ms,
            "model_inference_latency_ms": model_latency_ms,
        }

    def _global_cells_from_costmap(self, costmap, north_m, east_m, yaw_deg):
        """Project inflated occupied costmap cells into the global A* grid."""
        yaw_rad = math.radians(yaw_deg)
        cells = set()
        for y in range(costmap.height):
            for x in range(costmap.width):
                offset = costmap.index(x, y)
                if costmap.unknown[offset] or costmap.occupancy[offset] < 0.55:
                    continue
                forward_m = costmap.origin_forward_m + (x + 0.5) * costmap.resolution_m
                left_m = costmap.origin_left_m + (y + 0.5) * costmap.resolution_m
                obstacle_north = (
                    north_m
                    + forward_m * math.cos(yaw_rad)
                    - left_m * math.sin(yaw_rad)
                )
                obstacle_east = (
                    east_m
                    + forward_m * math.sin(yaw_rad)
                    + left_m * math.cos(yaw_rad)
                )
                cells.add(
                    (
                        int(math.floor(obstacle_east / self.resolution_m)),
                        int(math.floor(obstacle_north / self.resolution_m)),
                    )
                )
        return sorted(cells)

    def _recommended_direction(self, scan):
        sectors = {"left": [], "center": [], "right": []}
        for index, distance in enumerate(scan.ranges_m):
            if not math.isfinite(distance):
                continue
            angle_deg = math.degrees(scan.angle_at(index))
            if 15 <= angle_deg <= 60:
                sectors["left"].append(distance)
            elif -15 < angle_deg < 15:
                sectors["center"].append(distance)
            elif -60 <= angle_deg <= -15:
                sectors["right"].append(distance)
        clearance = {
            name: sum(values) / len(values) if values else 0.0
            for name, values in sectors.items()
        }
        return {"left": -35.0, "center": 0.0, "right": 35.0}[
            max(clearance, key=clearance.get)
        ]

    def _unhealthy_detection(self, reason, health, started):
        latency_ms = (time.perf_counter() - started) * 1000.0
        self.last_risk = RiskEstimate(
            level="danger",
            confidence=1.0,
            nearest_distance_m=None,
            collision_time_s=None,
            recommended_direction_deg=None,
            inference_latency_ms=latency_ms,
            model_id="sensor_watchdog_v1",
            reason=reason,
        )
        return {
            "detected_obstacle": False,
            "detected": False,
            "risk_level": "danger",
            "closest_obstacle_name": "",
            "closest_obstacle_distance_m": None,
            "closest_obstacle_bearing_deg": None,
            "closest_obstacle_in_detection_range": False,
            "closest_obstacle_in_fov": False,
            "nearest_obstacle_name": "",
            "nearest_obstacle_layer": "",
            "nearest_obstacle_distance_m": None,
            "nearest_obstacle_bearing_deg": None,
            "detected_obstacle_count": 0,
            "warning_distance_m": self.warning_distance_m,
            "danger_distance_m": self.danger_distance_m,
            "closest_obstacle": None,
            "nearest_obstacle": None,
            "detected_obstacles": [],
            "dynamic_grid_cells": [],
            "sensor_source": self.source.source_id,
            "sensor_healthy": False,
            "sensor_message": reason,
            "sensor_frequency_hz": health.frequency_hz,
            "sensor_frame_age_s": health.last_frame_age_s,
            "sensor_dropped_frames": health.dropped_frames,
            "costmap": None,
            "risk_estimate": self.last_risk,
            "risk_model_id": self.last_risk.model_id,
            "inference_latency_ms": latency_ms,
        }

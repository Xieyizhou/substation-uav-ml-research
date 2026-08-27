"""Live Gazebo RGB to active-inspection observation bridge.

Depth is intentionally an injected, synchronized provider. The bridge never
guesses range from a monocular bounding box and never reads map truth.
"""

from __future__ import annotations

from dataclasses import asdict, is_dataclass
from typing import Any, Callable, Mapping
from src.vision.localization.rgbd_localization import localize_bbox


class ActiveInspectionLiveBridge:
    def __init__(
        self,
        detector,
        temporal_update: Callable[[list[dict[str, Any]], float], list[dict[str, Any]]],
        depth_provider: Callable[[Any, Mapping[str, Any]], float | None],
        image_loader: Callable[[Any], Any],
        camera_intrinsics: Mapping[str, float],
        camera_extrinsics: Mapping[str, float] | None = None,
    ):
        self.detector = detector
        self.temporal_update = temporal_update
        self.depth_provider = depth_provider
        self.image_loader = image_loader
        self.camera_intrinsics = dict(camera_intrinsics)
        self.camera_extrinsics = dict(camera_extrinsics or {})
        if float(self.camera_intrinsics.get("fx", 0)) <= 0:
            raise ValueError("active inspection requires positive camera fx")
        self.last_audit = None

    @staticmethod
    def _detection_row(detection) -> dict[str, Any]:
        value = asdict(detection) if is_dataclass(detection) else dict(detection)
        bbox = value.get("bbox", value.get("bbox_xyxy"))
        return {
            "tracking_id": value.get("tracking_id"),
            "class_name": value["class_name"],
            "confidence": float(value["confidence"]),
            "bbox": list(bbox),
            "held": False,
        }

    def record(self, frame, vehicle_pose: Mapping[str, float], *, elapsed_s: float):
        timestamp_s = float(getattr(frame, "capture_timestamp", elapsed_s))
        detections = self.detector.detect(
            self.image_loader(frame), timestamp_s=timestamp_s,
            frame_id=str(getattr(frame, "frame_id", "gazebo_rgb")),
        )
        tracked = self.temporal_update(
            [self._detection_row(item) for item in detections], timestamp_s,
        )
        observations = []
        rejected = {
            "held": 0,
            "unstable": 0,
            "truncated_bbox": 0,
            "invalid_depth": 0,
        }
        for row in tracked:
            if row.get("held"):
                rejected["held"] += 1
                continue
            if not row.get("stable", True):
                rejected["unstable"] += 1
                continue
            width = getattr(frame, "width", None)
            height = getattr(frame, "height", None)
            if width is not None and height is not None:
                x1, y1, x2, y2 = map(float, row["bbox"])
                if x1 <= 0 or y1 <= 0 or x2 >= float(width) or y2 >= float(height):
                    rejected["truncated_bbox"] += 1
                    continue
            depth_m = self.depth_provider(frame, row)
            if depth_m is None or not .2 <= float(depth_m) <= 100:
                rejected["invalid_depth"] += 1
                continue
            observations.append({
                "tracking_id": row.get("tracking_id"),
                "class_name": row["class_name"],
                "confidence": float(row["confidence"]),
                "bbox": list(row["bbox"]),
                "held": False,
                "stable": True,
                "depth_m": float(depth_m),
                "camera_intrinsics": self.camera_intrinsics,
                "localized_position": localize_bbox(depth_m, row["bbox"], self.camera_intrinsics, vehicle_pose, self.camera_extrinsics),
            })
        self.last_audit = {
            "timestamp_s": timestamp_s,
            "raw_detections": [self._detection_row(item) for item in detections],
            "temporal_observations": [dict(row) for row in tracked],
            "accepted_observation_count": len(observations),
            "rejected": rejected,
        }
        if not observations:
            return None
        return {
            "trigger": "stable_detection",
            "timestamp_s": timestamp_s,
            "elapsed_s": float(elapsed_s),
            "vehicle_pose": dict(vehicle_pose),
            "observations": observations,
        }


async def gazebo_observation_records(
    camera, bridge: ActiveInspectionLiveBridge, pose_provider, *, timeout_s: float,
):
    """Yield truth-blind planner records from valid Gazebo RGB frames."""
    async for event in camera.events(timeout_s=timeout_s):
        if not event.valid:
            continue
        pose, elapsed_s = pose_provider(event.frame)
        record = bridge.record(event.frame, pose, elapsed_s=elapsed_s)
        if record is not None:
            yield record

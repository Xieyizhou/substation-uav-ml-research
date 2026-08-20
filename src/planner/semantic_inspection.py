"""Convert stable visual-depth observations into gated inspection plans."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import math

from src.maps.route_quality import evaluate_route_quality
from src.maps.sandbox_contracts import SandboxMap, SandboxMission
from src.maps.sandbox_routes import build_sandbox_route
from src.ml.artifacts import object_sha256


@dataclass(frozen=True)
class CameraIntrinsics:
    width_px: int
    height_px: int
    fx_px: float
    fy_px: float
    cx_px: float
    cy_px: float

    def __post_init__(self):
        if min(self.width_px, self.height_px) <= 0:
            raise ValueError("camera dimensions must be positive")
        if min(self.fx_px, self.fy_px) <= 0:
            raise ValueError("camera focal lengths must be positive")


@dataclass(frozen=True)
class VehiclePose:
    east_m: float
    north_m: float
    altitude_m: float
    yaw_deg: float


@dataclass(frozen=True)
class SemanticEquipmentEstimate:
    class_name: str
    confidence: float
    east_m: float
    north_m: float
    altitude_m: float
    depth_m: float
    source_tracking_id: str

    @property
    def estimate_identity_sha256(self):
        return object_sha256(asdict(self))


@dataclass(frozen=True)
class SemanticInspectionPlan:
    estimate: SemanticEquipmentEstimate
    matched_object_id: str
    localization_error_m: float
    mission: SandboxMission
    route: dict
    route_quality: dict

    @property
    def plan_identity_sha256(self):
        return object_sha256(asdict(self))


def localize_equipment(observation, depth_m, intrinsics, pose):
    if not observation.get("tracking_id"):
        raise ValueError("semantic planning requires a stable tracking ID")
    if observation.get("held"):
        raise ValueError("held observations cannot initialize a spatial estimate")
    depth_m = float(depth_m)
    if not 0.2 <= depth_m <= 100.0:
        raise ValueError("depth must be between 0.2 and 100 m")
    left, top, right, bottom = map(float, observation["bbox"])
    center_x, center_y = (left + right) / 2.0, (top + bottom) / 2.0
    horizontal_angle = math.atan2(center_x - intrinsics.cx_px, intrinsics.fx_px)
    vertical_angle = math.atan2(intrinsics.cy_px - center_y, intrinsics.fy_px)
    yaw = math.radians(pose.yaw_deg) + horizontal_angle
    horizontal_depth = depth_m * math.cos(vertical_angle)
    return SemanticEquipmentEstimate(
        class_name=str(observation["class_name"]),
        confidence=float(observation["confidence"]),
        east_m=pose.east_m + math.sin(yaw) * horizontal_depth,
        north_m=pose.north_m + math.cos(yaw) * horizontal_depth,
        altitude_m=pose.altitude_m + math.sin(vertical_angle) * depth_m,
        depth_m=depth_m,
        source_tracking_id=str(observation["tracking_id"]),
    )


def _matching_object(map_value, estimate, tolerance_m):
    candidates = [
        (
            math.hypot(item.east_m - estimate.east_m, item.north_m - estimate.north_m),
            item,
        )
        for item in map_value.objects
        if item.asset_id == estimate.class_name and item.label_role == "target"
    ]
    distance, item = min(candidates, default=(math.inf, None), key=lambda row: row[0])
    if item is None:
        raise ValueError(f"map has no target compatible with {estimate.class_name}")
    if distance > tolerance_m:
        raise ValueError("semantic estimate does not match a mapped target")
    return item, distance


def plan_semantic_inspection(
    map_value,
    estimate,
    *,
    localization_tolerance_m=3.0,
    altitude_m=1.5,
    speed_m_s=1.0,
):
    target, error = _matching_object(
        map_value, estimate, float(localization_tolerance_m)
    )
    mission = SandboxMission(
        mission_id=f"inspect_{target.object_id}",
        mission_type="equipment_inspection",
        target_object_id=target.object_id,
        altitude_m=altitude_m,
        speed_m_s=speed_m_s,
    )
    route = build_sandbox_route(map_value, mission)
    quality = evaluate_route_quality(map_value, mission, route)
    if not quality.accepted:
        codes = ", ".join(item.code for item in quality.issues)
        raise ValueError(f"semantic inspection route failed quality gate: {codes}")
    return SemanticInspectionPlan(
        estimate,
        target.object_id,
        round(error, 6),
        mission,
        route.to_record(),
        quality.to_record(),
    )

"""Simulated perception state construction and detector setup."""

from src.perception.perception_state import build_perception_state
from src.perception.simple_obstacle_detector import SimpleObstacleDetector
from src.perception.lidar_detector import LidarRiskDetector
from src.sensors.factory import build_lidar_source

from src.flight.flight_state import value_or_blank


class DangerObstacleDetected(Exception):
    """Raised when the configured risk action requires an immediate landing."""


class SensorDataUnavailable(Exception):
    """Raised when a required real-time perception stream is stale or unavailable."""


def current_perception_detection(
    perception_config,
    detector,
    position,
    attitude,
    timestamp_utc=None,
    elapsed_s=None,
    replan_config=None,
    velocity=None,
):
    enabled = bool(perception_config.get("enabled"))
    if not enabled or detector is None or position is None:
        return build_perception_state(
            perception_config,
            detection=None,
            position=position,
            attitude=attitude,
            timestamp_utc=timestamp_utc,
            elapsed_s=elapsed_s,
            replan_config=replan_config,
        )
    yaw_deg = value_or_blank(attitude, "yaw_deg")
    altitude_m = -position.down_m if hasattr(position, "down_m") else ""
    detection = detector.detect(
        local_north_m=position.north_m,
        local_east_m=position.east_m,
        yaw_deg=yaw_deg,
        altitude_m=altitude_m,
        velocity_ned_m_s=(
            (
                value_or_blank(velocity, "north_m_s") or 0.0,
                value_or_blank(velocity, "east_m_s") or 0.0,
                value_or_blank(velocity, "down_m_s") or 0.0,
            )
            if velocity is not None
            else None
        ),
    )
    return build_perception_state(
        perception_config,
        detection=detection,
        position=position,
        attitude=attitude,
        timestamp_utc=timestamp_utc,
        elapsed_s=elapsed_s,
        replan_config=replan_config,
    )


def build_perception_detector(args, planner_config):
    if not args.enable_perception:
        return None
    if args.perception_source == "map_baseline":
        if (
            planner_config.get("obstacle_config") is None
            or planner_config.get("obstacle_map") is None
        ):
            raise ValueError("map baseline perception requires --obstacle-config")
        return SimpleObstacleDetector(
            planner_config["obstacle_config"],
            obstacle_map=planner_config["obstacle_map"],
            detection_range_m=args.detection_range,
            warning_distance_m=args.warning_distance,
            danger_distance_m=args.danger_distance,
            detection_fov_deg=args.detection_fov,
            use_inflated_cells=args.perception_use_inflated,
            use_raw_cells=args.perception_use_raw,
            flight_altitude_m=planner_config["altitude_m"],
        )
    source = build_lidar_source(
        args.perception_source,
        topic=args.sensor_topic,
        replay_path=args.sensor_replay,
        stale_after_s=args.sensor_stale_after,
        scenario_manifest=args.scenario_manifest,
    )
    risk_predictor = None
    if args.risk_model != "geometric":
        from src.ml.onnx_risk import OnnxRiskModel

        risk_predictor = OnnxRiskModel(args.risk_model)
    return LidarRiskDetector(
        source,
        resolution_m=planner_config["resolution_m"],
        detection_range_m=args.detection_range,
        warning_distance_m=args.warning_distance,
        danger_distance_m=args.danger_distance,
        detection_fov_deg=args.detection_fov,
        stale_after_s=args.sensor_stale_after,
        nominal_speed_m_s=args.max_speed,
        inflation_radius_m=(
            planner_config.get("horizontal_inflation_cells", 1)
            * planner_config["resolution_m"]
        ),
        risk_predictor=risk_predictor,
        risk_fusion=args.risk_fusion,
    )

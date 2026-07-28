"""Runtime validation plus perception and replanning configuration."""

def validate_runtime_args(args):
    """Reject unsafe or internally inconsistent runtime settings."""
    positive_fields = {
        "--max-speed": args.max_speed,
        "--return-speed-scale": args.return_speed_scale,
        "--waypoint-acceptance": args.waypoint_acceptance,
        "--min-risk-speed": args.min_risk_speed,
        "--resolution": args.resolution,
        "--detection-range": args.detection_range,
        "--warning-distance": args.warning_distance,
        "--danger-distance": args.danger_distance,
        "--connection-timeout": args.connection_timeout,
        "--position-ready-timeout": args.position_ready_timeout,
        "--telemetry-timeout": args.telemetry_timeout,
        "--landing-timeout": args.landing_timeout,
        "--logger-shutdown-timeout": args.logger_shutdown_timeout,
        "--sensor-startup-timeout": getattr(args, "sensor_startup_timeout", 5.0),
        "--sensor-stale-after": getattr(args, "sensor_stale_after", 0.5),
    }
    if args.altitude is not None:
        positive_fields["--altitude"] = args.altitude
    for flag, value in positive_fields.items():
        if value <= 0:
            raise ValueError(f"{flag} must be positive")

    if args.turn_settle < 0:
        raise ValueError("--turn-settle must be non-negative")
    if not 0 < args.detection_fov <= 360:
        raise ValueError("--detection-fov must be greater than 0 and at most 360")
    if not args.danger_distance <= args.warning_distance <= args.detection_range:
        raise ValueError(
            "Perception distances must satisfy danger <= warning <= detection range"
        )
    if (
        args.enable_perception
        and args.risk_action == "slow_down"
        and args.min_risk_speed > args.max_speed
    ):
        raise ValueError("--min-risk-speed must not exceed --max-speed")
    perception_source = getattr(args, "perception_source", "map_baseline")
    sensor_replay = getattr(args, "sensor_replay", None)
    if perception_source == "replay" and sensor_replay is None:
        raise ValueError("--sensor-replay is required with --perception-source replay")
    if perception_source != "map_baseline" and not args.enable_perception:
        raise ValueError("a real sensor source requires --enable-perception")

def build_perception_config(args):
    """Return perception settings consumed by the detector, logger, and flight loop."""
    return {
        "enabled": args.enable_perception,
        "detector_name": (
            "simple_obstacle_detector"
            if args.perception_source == "map_baseline"
            else "lidar_risk_detector"
        ),
        "source": args.perception_source,
        "sensor_topic": args.sensor_topic,
        "sensor_replay": str(args.sensor_replay) if args.sensor_replay else None,
        "sensor_startup_timeout_s": args.sensor_startup_timeout,
        "sensor_stale_after_s": args.sensor_stale_after,
        "risk_model": args.risk_model,
        "equipment_model": str(args.equipment_model) if args.equipment_model else None,
        "detection_range_m": args.detection_range,
        "detection_fov_deg": args.detection_fov,
        "warning_distance_m": args.warning_distance,
        "danger_distance_m": args.danger_distance,
        "risk_action": args.risk_action,
        "use_raw_cells": args.perception_use_raw,
        "use_inflated_cells": args.perception_use_inflated,
    }


def print_perception_summary(perception_config):
    """Print the active perception configuration before preview or flight."""
    print("\nPerception:")
    print(f"  enabled: {str(perception_config['enabled']).lower()}")
    print(f"  detector: {perception_config['detector_name']}")
    print(f"  source: {perception_config['source']}")
    print(f"  risk model: {perception_config['risk_model']}")
    print(f"  detection range: {perception_config['detection_range_m']} m")
    print(f"  detection FOV: {perception_config['detection_fov_deg']} deg")
    print(f"  warning distance: {perception_config['warning_distance_m']} m")
    print(f"  danger distance: {perception_config['danger_distance_m']} m")
    print(f"  risk action: {perception_config['risk_action']}")
    print(f"  uses raw cells: {str(perception_config['use_raw_cells']).lower()}")
    print(f"  uses inflated cells: {str(perception_config['use_inflated_cells']).lower()}")


def build_replan_config(args, planner_config):
    """Return local-replan settings derived from CLI arguments and map metadata.

    The returned dictionary is intentionally plain data so `fly_astar_path.py`
    can log it, test trigger thresholds, and call A* without reaching back into
    argparse or obstacle-config internals.
    """
    if args.replan_cooldown < 0:
        raise ValueError("--replan-cooldown must be non-negative")
    if args.dynamic_obstacle_inflation < 0:
        raise ValueError("--dynamic-obstacle-inflation must be non-negative")
    if args.max_replans < 0:
        raise ValueError("--max-replans must be non-negative")

    enabled = bool(args.enable_local_replan)
    if enabled and not args.enable_perception:
        raise ValueError("--enable-local-replan requires --enable-perception")
    if enabled and planner_config.get("obstacle_map") is None:
        raise ValueError("--enable-local-replan requires --obstacle-config")

    return {
        "enabled": enabled,
        "mode": args.replan_mode,
        "risk_level": args.replan_risk_level,
        "cooldown_s": args.replan_cooldown,
        "dynamic_obstacle_inflation": args.dynamic_obstacle_inflation,
        "max_replans": args.max_replans,
        "width": planner_config["width"],
        "height": planner_config["height"],
        "resolution_m": planner_config["resolution_m"],
        "altitude_m": planner_config["altitude_m"],
        "goal_cell": planner_config["goal"],
        "static_obstacles": set(planner_config["inflated_blocking_cells"]),
        "allow_diagonal": args.allow_diagonal,
    }


def print_replan_summary(replan_config):
    """Print the active local-replan configuration before preview or flight."""
    print("\nLocal replanning:")
    print(f"  enabled: {str(replan_config['enabled']).lower()}")
    print(f"  mode: {replan_config['mode']}")
    print(f"  trigger risk level: {replan_config['risk_level']}")
    print(f"  cooldown: {replan_config['cooldown_s']} s")
    print(f"  dynamic obstacle inflation: {replan_config['dynamic_obstacle_inflation']} cell(s)")
    print(f"  max replans: {replan_config['max_replans']}")
    print(f"  active waypoint replacement: {str(replan_config['mode'] == 'active').lower()}")

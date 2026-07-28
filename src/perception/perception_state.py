"""Structured state helpers for simulated real-time perception."""

RISK_LEVEL_ORDER = {
    "clear": 0,
    "detected": 1,
    "warning": 2,
    "danger": 3,
}


def _value_or_none(message, attribute):
    if message is None:
        return None
    value = getattr(message, attribute, None)
    return value if value != "" else None


def _position_state(position):
    return {
        "north_m": _value_or_none(position, "north_m"),
        "east_m": _value_or_none(position, "east_m"),
        "down_m": _value_or_none(position, "down_m"),
    }


def _heading_state(attitude):
    return {
        "yaw_deg": _value_or_none(attitude, "yaw_deg"),
    }


def risk_value(risk_level):
    return RISK_LEVEL_ORDER.get(risk_level, 0)


def risk_reaches_threshold(risk_level, threshold):
    return risk_value(risk_level) >= RISK_LEVEL_ORDER.get(threshold, 3)


def suggested_action_for_state(perception_config, risk_level, replan_config=None):
    if risk_level == "clear":
        return "none"

    if replan_config and replan_config.get("enabled"):
        if risk_reaches_threshold(risk_level, replan_config.get("risk_level", "danger")):
            return "replan_candidate"

    risk_action = perception_config.get("risk_action", "log_only")
    if risk_action == "stop_and_land" and risk_level == "danger":
        return "stop_and_land"
    if risk_action == "slow_down" and risk_level in {"warning", "danger"}:
        return "slow_down"
    if risk_action == "log_only":
        return "log_only"
    return "none"


def risk_reason_for_state(enabled, detection, closest_obstacle, risk_level):
    if not enabled:
        return "perception disabled"
    if detection is None:
        return "local position unavailable"
    if risk_level == "danger":
        return "closest detected obstacle is inside danger distance"
    if risk_level == "warning":
        return "closest detected obstacle is inside warning distance"
    if risk_level == "detected":
        return "obstacle is inside detection range and forward FOV"
    if closest_obstacle is None:
        return "no obstacle available in perception map"
    if not closest_obstacle.get("in_detection_range"):
        return "closest obstacle is outside detection range"
    if not closest_obstacle.get("in_fov"):
        return "closest obstacle is outside forward FOV"
    return "no obstacle inside detection range and forward FOV"


def build_perception_state(
    perception_config,
    detection=None,
    position=None,
    attitude=None,
    timestamp_utc=None,
    elapsed_s=None,
    replan_config=None,
):
    """Return a stable perception state dict while preserving legacy detection keys."""
    enabled = bool(perception_config.get("enabled"))
    risk_level = detection.get("risk_level", "clear") if detection else "clear"
    nearest_obstacle = detection.get("nearest_obstacle") if detection else None
    closest_obstacle = detection.get("closest_obstacle") if detection else None

    state = {
        "enabled": enabled,
        "timestamp_utc": timestamp_utc,
        "elapsed_s": elapsed_s,
        "position": _position_state(position),
        "heading": _heading_state(attitude),
        "closest_obstacle": closest_obstacle,
        "closest_obstacle_name": (
            closest_obstacle.get("obstacle_name", "") if closest_obstacle else ""
        ),
        "closest_obstacle_distance_m": (
            closest_obstacle.get("distance_m") if closest_obstacle else None
        ),
        "in_detection_range": bool(
            closest_obstacle.get("in_detection_range") if closest_obstacle else False
        ),
        "in_fov": bool(closest_obstacle.get("in_fov") if closest_obstacle else False),
        "risk_level": risk_level,
        "risk_value": risk_value(risk_level),
        "risk_reason": risk_reason_for_state(
            enabled,
            detection,
            closest_obstacle,
            risk_level,
        ),
        "suggested_action": (
            suggested_action_for_state(perception_config, risk_level, replan_config)
            if enabled
            else "none"
        ),
    }

    if detection:
        state.update(detection)
    else:
        state.update(
            {
                "detected_obstacle": False,
                "nearest_obstacle_name": "",
                "nearest_obstacle_layer": "",
                "nearest_obstacle_distance_m": None,
                "nearest_obstacle_bearing_deg": None,
                "detected_obstacle_count": 0,
                "detected": False,
                "nearest_obstacle": None,
                "detected_obstacles": [],
            }
        )
    state["risk_value"] = risk_value(state.get("risk_level", "clear"))
    return state

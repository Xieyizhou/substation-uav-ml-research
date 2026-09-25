"""Waypoint value policies and diagnostics, independent of control-loop state."""

from src.flight.flight_state import horizontal_distance_to_waypoint


def clamp(value, min_value, max_value):
    return max(min_value, min(max_value, value))


def normalize_yaw_deg(yaw_deg):
    normalized = (float(yaw_deg) + 180.0) % 360.0 - 180.0
    return 180.0 if normalized == -180.0 else normalized


def return_has_geometric_obstacle_evidence(route_direction, detection):
    evidence_fields = ("nearest_obstacle", "detected_obstacles", "dynamic_grid_cells")
    return route_direction == "return" and bool(detection) and any(detection.get(field) for field in evidence_fields)


def print_waypoint_timeout_info(waypoint, timeout_info):
    print(f"Flying to {waypoint['name']}...")
    print(f"  distance: {timeout_info['distance_m']:.2f} m")
    print(f"  expected speed: {timeout_info['expected_speed_m_s']:.2f} m/s")
    print(f"  timeout: {timeout_info['timeout_s']:.1f} s")


def timeout_info(position, waypoint, speed_scale, perception_config,
                 max_horizontal_speed, min_risk_speed, timeout_mode):
    distance_m = horizontal_distance_to_waypoint(position, waypoint) or 0.0
    expected_speed_m_s = max_horizontal_speed * speed_scale
    if perception_config.get("risk_action", "log_only") == "slow_down":
        expected_speed_m_s = max(expected_speed_m_s * 0.35, min_risk_speed)
    if timeout_mode == "auto":
        base_timeout_s = distance_m / max(expected_speed_m_s, 0.2)
        timeout_s = max(20.0, base_timeout_s * 3.0 + 10.0)
        if perception_config.get("risk_action", "log_only") == "slow_down":
            timeout_s *= 2.0
    else:
        timeout_s = timeout_mode
    return {
        "distance_m": distance_m,
        "expected_speed_m_s": expected_speed_m_s,
        "timeout_s": timeout_s,
    }


def print_timeout_debug(waypoint, position, error, detection, command_speed):
    nearest = detection["nearest_obstacle"] if detection else None
    print("Waypoint timeout debug:")
    print(f"  waypoint: {waypoint['name']}")
    print(
        "  target N/E/D: "
        f"{waypoint['north_m']:.2f}, {waypoint['east_m']:.2f}, {waypoint['down_m']:.2f}"
    )
    if position is None:
        print("  latest local N/E/D: unavailable")
        print("  current horizontal error: unavailable")
    else:
        print(
            "  latest local N/E/D: "
            f"{position.north_m:.2f}, {position.east_m:.2f}, {position.down_m:.2f}"
        )
        print(f"  current horizontal error: {error['horizontal_m']:.2f} m")
    print(f"  latest perception_risk_level: {detection['risk_level'] if detection else 'clear'}")
    if nearest is None:
        print("  latest nearest obstacle: unavailable")
    else:
        print(
            f"  latest nearest obstacle: {nearest['obstacle_name']} "
            f"at {nearest['distance_m']:.2f} m"
        )
    print(
        "  commanded speed: unavailable"
        if command_speed is None
        else f"  commanded speed: {command_speed:.2f} m/s"
    )



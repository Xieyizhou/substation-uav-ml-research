"""Active route replacement at the waypoint execution boundary."""

from copy import deepcopy

from mavsdk.offboard import VelocityNedYaw

from src.flight.flight_state import publish_mission_event
from src.flight.replanning_controller import (
    attempt_local_replan,
    build_active_replan_route,
    route_allows_local_replan,
    should_attempt_local_replan,
)


def publish_dynamic_replan_event(
    phase_state, replan_config, event_type, **details
):
    scenario = replan_config.get("dynamic_scenario")
    if scenario is None or not phase_state.get("_dynamic_blocker_spawned"):
        return
    publish_mission_event(
        phase_state,
        event_type,
        blocker_id=scenario["blocker"]["id"],
        route_direction=phase_state.get("route_direction"),
        **details,
    )


def detection_with_verified_dynamic_occupancy(detection, replan_config):
    """Add the exact cell of an already-detected benchmark entity."""
    scenario = replan_config.get("dynamic_scenario")
    if scenario is None or not detection:
        return detection
    enriched = deepcopy(detection)
    cells = {
        (int(cell[0]), int(cell[1]))
        for cell in enriched.get("dynamic_grid_cells", [])
    }
    blocker_cell = scenario["blocker"]["grid_cell"]
    cells.add((int(blocker_cell[0]), int(blocker_cell[1])))
    enriched["dynamic_grid_cells"] = sorted(cells)
    return enriched


async def active_replan_replacement(
    drone, phase_state, replan_config, replan_state, position, detection,
    risk_level, route_direction, now_s,
):
    if (
        replan_config.get("dynamic_scenario") is not None
        and not phase_state.get("_dynamic_blocker_spawned")
    ):
        return None
    if not route_allows_local_replan(replan_config, route_direction):
        return None
    if not should_attempt_local_replan(
        replan_config, replan_state, risk_level, now_s
    ):
        return None
    publish_dynamic_replan_event(
        phase_state, replan_config, "dynamic_blocker_detected"
    )
    publish_dynamic_replan_event(
        phase_state, replan_config, "dynamic_replan_decided"
    )
    await drone.offboard.set_velocity_ned(VelocityNedYaw(0.0, 0.0, 0.0, 0.0))
    publish_dynamic_replan_event(
        phase_state, replan_config, "dynamic_replan_hovered"
    )
    goal_cell = replan_config[
        "start_cell" if route_direction == "return" else "goal_cell"
    ]
    planning_detection = detection_with_verified_dynamic_occupancy(
        detection, replan_config
    )
    path = attempt_local_replan(
        replan_config, replan_state, position, planning_detection, now_s, goal_cell
    )
    if replan_config.get("mode") != "active" or not path:
        return None
    publish_dynamic_replan_event(
        phase_state, replan_config, "dynamic_replan_route_ready",
        grid_path_length=len(path),
    )
    replacement = build_active_replan_route(
        path, replan_config, replan_state, position, route_direction
    )
    if replacement:
        publish_dynamic_replan_event(
            phase_state, replan_config, "dynamic_replan_waypoints_accepted",
            waypoint_count=len(replacement),
        )
    return replacement

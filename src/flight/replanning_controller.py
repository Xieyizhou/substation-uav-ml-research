"""Risk-triggered local A* replanning and active route replacement."""

from math import floor

from src.perception.perception_state import risk_reaches_threshold
from src.planner.astar_grid import astar, grid_path_to_local_waypoints, simplify_grid_path
from src.planner.obstacle_config import inflate_cells


REACHED_HORIZONTAL_ERROR_M = 0.4
REACHED_VERTICAL_ERROR_M = 0.4


def configure_acceptance(horizontal_m, vertical_m):
    global REACHED_HORIZONTAL_ERROR_M, REACHED_VERTICAL_ERROR_M
    REACHED_HORIZONTAL_ERROR_M = horizontal_m
    REACHED_VERTICAL_ERROR_M = vertical_m


def local_position_to_grid_cell(position, resolution_m):
    if position is None or resolution_m <= 0:
        return None
    return (
        int(floor(position.east_m / resolution_m)),
        int(floor(position.north_m / resolution_m)),
    )


def empty_replan_state():
    return {
        "replan_mode": "log_only",
        "replan_triggered": False,
        "replan_success": "",
        "replan_start_grid_x": "",
        "replan_start_grid_y": "",
        "replan_goal_grid_x": "",
        "replan_goal_grid_y": "",
        "replan_path_length": "",
        "dynamic_blocked_cell_count": "",
        "replan_count": 0,
        "replan_route_replaced": False,
        "active_replan_count": 0,
        "active_replan_path_length": "",
        "following_replanned_route": False,
        "last_attempt_time": None,
    }


def reset_replan_event_fields(replan_state):
    for field in (
        "replan_success",
        "replan_start_grid_x",
        "replan_start_grid_y",
        "replan_goal_grid_x",
        "replan_goal_grid_y",
        "replan_path_length",
        "dynamic_blocked_cell_count",
        "active_replan_path_length",
    ):
        replan_state[field] = ""
    replan_state["replan_triggered"] = False
    replan_state["replan_route_replaced"] = False


def dynamic_cells_from_detection(detection):
    if not detection:
        return set()
    explicit_cells = detection.get("dynamic_grid_cells")
    if explicit_cells is not None:
        return {
            (int(cell[0]), int(cell[1]))
            for cell in explicit_cells
            if isinstance(cell, (list, tuple)) and len(cell) == 2
        }
    cells = set()
    for obstacle in detection.get("detected_obstacles", []):
        grid_x = obstacle.get("grid_x")
        grid_y = obstacle.get("grid_y")
        if grid_x is not None and grid_y is not None:
            cells.add((int(grid_x), int(grid_y)))
    return cells


def should_attempt_local_replan(replan_config, replan_state, risk_level, now_s):
    if not replan_config.get("enabled"):
        return False
    if replan_state.get("replan_count", 0) >= replan_config["max_replans"]:
        return False
    if replan_state.get("following_replanned_route"):
        return False
    if not risk_reaches_threshold(risk_level, replan_config["risk_level"]):
        return False
    last_attempt_time = replan_state.get("last_attempt_time")
    return not (
        last_attempt_time is not None
        and now_s - last_attempt_time < replan_config["cooldown_s"]
    )


def route_allows_local_replan(replan_config, route_direction):
    return (
        replan_config.get("mode") != "active"
        or route_direction == "outbound"
        or (
            route_direction == "return"
            and replan_config.get("allow_return_replan", False)
        )
    )


def update_replanned_route_escape(replan_config, replan_state, risk_level):
    """Keep the escape route active until its triggering risk has cleared."""
    following = bool(replan_state.get("following_replanned_route"))
    if following and not risk_reaches_threshold(
        risk_level, replan_config["risk_level"]
    ):
        replan_state["following_replanned_route"] = False
        return False
    return following


def attempt_local_replan(
    replan_config, replan_state, position, detection, now_s, goal_cell=None
):
    replan_state["last_attempt_time"] = now_s
    replan_state["replan_count"] = replan_state.get("replan_count", 0) + 1
    reset_replan_event_fields(replan_state)
    replan_state["replan_triggered"] = True
    start_cell = local_position_to_grid_cell(position, replan_config["resolution_m"])
    goal_cell = goal_cell or replan_config["goal_cell"]
    if start_cell is not None:
        replan_state["replan_start_grid_x"] = start_cell[0]
        replan_state["replan_start_grid_y"] = start_cell[1]
    replan_state["replan_goal_grid_x"] = goal_cell[0]
    replan_state["replan_goal_grid_y"] = goal_cell[1]
    dynamic_cells = dynamic_cells_from_detection(detection)
    dynamic_cells -= set(replan_config["static_obstacles"])
    inflated_dynamic_cells = inflate_cells(
        dynamic_cells,
        replan_config["width"],
        replan_config["height"],
        replan_config["dynamic_obstacle_inflation"],
    )
    replan_state["dynamic_blocked_cell_count"] = len(inflated_dynamic_cells)
    if start_cell is None:
        replan_state["replan_success"] = "false"
        print("Local replan attempt failed: current local position is unavailable.")
        return None
    planning_obstacles = set(replan_config["static_obstacles"]) | inflated_dynamic_cells
    planning_obstacles -= {start_cell, goal_cell}
    try:
        replanned_path = astar(
            start=start_cell,
            goal=goal_cell,
            obstacles=planning_obstacles,
            width=replan_config["width"],
            height=replan_config["height"],
            allow_diagonal=replan_config["allow_diagonal"],
        )
    except ValueError as error:
        replan_state["replan_success"] = "false"
        print(
            "Local replan attempt failed: "
            f"start={start_cell}, goal={goal_cell}, dynamic_cells={len(inflated_dynamic_cells)}, "
            f"error={error}"
        )
        return None
    replan_state["replan_success"] = "true"
    replan_state["replan_path_length"] = len(replanned_path)
    print(
        "Local replan attempt succeeded: "
        f"start={start_cell}, goal={goal_cell}, dynamic_cells={len(inflated_dynamic_cells)}, "
        f"path_length={len(replanned_path)}"
    )
    return replanned_path


def replanned_path_to_waypoints(replanned_path, replan_config):
    simplified_path = simplify_grid_path(replanned_path)
    waypoints = grid_path_to_local_waypoints(
        simplified_path, replan_config["resolution_m"], replan_config["altitude_m"]
    )
    for index, waypoint in enumerate(waypoints, start=1):
        waypoint["name"] = f"RWP{index:02d}"
    return waypoints


def first_useful_replan_waypoint_index(position, waypoints):
    if position is None:
        return 0
    for index, waypoint in enumerate(waypoints):
        north_error = waypoint["north_m"] - position.north_m
        east_error = waypoint["east_m"] - position.east_m
        horizontal_error = (north_error**2 + east_error**2) ** 0.5
        vertical_error = waypoint["down_m"] - position.down_m
        if (
            horizontal_error >= REACHED_HORIZONTAL_ERROR_M
            or abs(vertical_error) >= REACHED_VERTICAL_ERROR_M
        ):
            return index
    return max(len(waypoints) - 1, 0)


def build_active_replan_route(
    replanned_path, replan_config, replan_state, position, route_direction="outbound"
):
    replanned_waypoints = replanned_path_to_waypoints(replanned_path, replan_config)
    if not replanned_waypoints:
        return None
    first_index = first_useful_replan_waypoint_index(position, replanned_waypoints)
    replacement_waypoints = replanned_waypoints[first_index:] or [replanned_waypoints[-1]]
    replan_state["replan_route_replaced"] = True
    replan_state["following_replanned_route"] = True
    replan_state["active_replan_count"] = replan_state.get("active_replan_count", 0) + 1
    replan_state["active_replan_path_length"] = len(replanned_path)
    print(
        f"Active local replan replaced {route_direction} route: "
        f"replacement_waypoints={len(replacement_waypoints)}, "
        f"grid_path_length={len(replanned_path)}, skipped_waypoints={first_index}"
    )
    return replacement_waypoints

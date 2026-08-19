"""Convert existing read-only catalog maps into sandbox editor templates."""

from __future__ import annotations

from src.maps.map_catalog import list_maps, project_path
from src.maps.sandbox_assets import ASSET_BY_ID, TARGET_ASSET_IDS
from src.maps.sandbox_contracts import SandboxMap, SandboxMapObject, SandboxMission
from src.planner.obstacle_config import load_obstacle_config


def _asset_id(obstacle):
    value = str(obstacle.get("visual_category", "generic_obstacle"))
    return value if value in ASSET_BY_ID else "generic_obstacle"


def _object(index, obstacle):
    if obstacle["type"] == "cell":
        x_min = x_max = int(obstacle["x"])
        y_min = y_max = int(obstacle["y"])
    else:
        x_min, x_max = int(obstacle["x_min"]), int(obstacle["x_max"])
        y_min, y_max = int(obstacle["y_min"]), int(obstacle["y_max"])
    asset_id = _asset_id(obstacle)
    return SandboxMapObject(
        str(obstacle.get("name") or f"object_{index:03d}"), asset_id,
        (x_min + x_max + 1) / 2.0, (y_min + y_max + 1) / 2.0,
        x_max - x_min + 1.0, y_max - y_min + 1.0,
        float(obstacle.get("z_max_m", 2.0)) - float(obstacle.get("z_min_m", 0.0)),
        0.0, "target" if asset_id in TARGET_ASSET_IDS else "background",
    )


def sandbox_template(entry):
    config = load_obstacle_config(project_path(entry["obstacle_config"]))
    start = config["start_cell"]
    goal = config["goal_cell"]
    return SandboxMap(
        f"template_{entry['id']}", entry["display_name"],
        float(config["width"]), float(config["height"]),
        start[0] + 0.5, start[1] + 0.5, float(entry["spawn_pose"][5]),
        tuple(_object(index, obstacle) for index, obstacle in enumerate(config["obstacles"], start=1)),
        (SandboxMission("default_round_trip", "round_trip", goal[0] + 0.5, goal[1] + 0.5),),
        weather="clear",
    )


def sandbox_templates():
    return tuple(sandbox_template(entry) for entry in list_maps())

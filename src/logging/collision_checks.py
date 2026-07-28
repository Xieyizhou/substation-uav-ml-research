"""Obstacle collision and safety-buffer validation helpers for offline A* analysis."""

import math


def obstacle_collision_report(df, obstacle_map, resolution_m):
    raw_cells = obstacle_map.get("raw_obstacle_cells", set()) if obstacle_map else set()
    inflated_cells = obstacle_map.get("inflated_blocking_cells", set()) if obstacle_map else set()
    raw_names_by_cell = obstacle_map.get("raw_obstacle_cell_to_name", {}) if obstacle_map else {}
    inflated_names_by_cell = obstacle_map.get("inflated_obstacle_cell_to_name", {}) if obstacle_map else {}
    report = {
        "raw_physical_collision_detected": False,
        "inflated_safety_buffer_entry_detected": False,
        "raw_collision_points": 0,
        "inflated_buffer_entry_points": 0,
        "first_raw_collision_timestamps": [],
        "first_inflated_buffer_entry_timestamps": [],
        "raw_obstacle_names_involved": [],
        "inflated_obstacle_names_involved": [],
        "approximate_min_clearance_m": None,
        "raw_collision_rows": [],
        "inflated_buffer_entry_rows": [],
        "collision_rows": [],
        "obstacle_collision_detected": False,
        "points_inside_obstacles": 0,
        "first_collision_timestamps": [],
    }
    if (not raw_cells and not inflated_cells) or not resolution_m:
        return report

    position_df = df[["elapsed_s", "local_north_m", "local_east_m"]].dropna()
    obstacle_centers = [
        ((x + 0.5) * resolution_m, (y + 0.5) * resolution_m)
        for x, y in inflated_cells
    ]
    min_clearance = None
    raw_timestamps = []
    inflated_timestamps = []
    raw_names = set()
    inflated_names = set()
    for _, row in position_df.iterrows():
        east = float(row["local_east_m"])
        north = float(row["local_north_m"])
        cell = (math.floor(east / resolution_m), math.floor(north / resolution_m))
        if cell in raw_cells:
            names = raw_names_by_cell.get(cell, "")
            raw_names.update(name for name in names.split(",") if name)
            row_data = {
                "entry_type": "raw_physical",
                "elapsed_s": float(row["elapsed_s"]),
                "local_east_m": east,
                "local_north_m": north,
                "grid_x": int(cell[0]),
                "grid_y": int(cell[1]),
                "obstacle_names": names,
            }
            raw_timestamps.append(float(row["elapsed_s"]))
            report["raw_collision_rows"].append(row_data)
            report["collision_rows"].append(row_data)
        if cell in inflated_cells:
            names = inflated_names_by_cell.get(cell, "")
            inflated_names.update(name for name in names.split(",") if name)
            row_data = {
                "entry_type": "inflated_buffer",
                "elapsed_s": float(row["elapsed_s"]),
                "local_east_m": east,
                "local_north_m": north,
                "grid_x": int(cell[0]),
                "grid_y": int(cell[1]),
                "obstacle_names": names,
            }
            inflated_timestamps.append(float(row["elapsed_s"]))
            report["inflated_buffer_entry_rows"].append(row_data)
            report["collision_rows"].append(row_data)
        for center_east, center_north in obstacle_centers:
            distance = math.hypot(east - center_east, north - center_north)
            if min_clearance is None or distance < min_clearance:
                min_clearance = distance

    report["raw_physical_collision_detected"] = bool(raw_timestamps)
    report["inflated_safety_buffer_entry_detected"] = bool(inflated_timestamps)
    report["raw_collision_points"] = len(raw_timestamps)
    report["inflated_buffer_entry_points"] = len(inflated_timestamps)
    report["first_raw_collision_timestamps"] = raw_timestamps[:5]
    report["first_inflated_buffer_entry_timestamps"] = inflated_timestamps[:5]
    report["raw_obstacle_names_involved"] = sorted(raw_names)
    report["inflated_obstacle_names_involved"] = sorted(inflated_names)
    report["approximate_min_clearance_m"] = min_clearance
    report["obstacle_collision_detected"] = bool(raw_timestamps or inflated_timestamps)
    report["points_inside_obstacles"] = len(inflated_timestamps)
    report["first_collision_timestamps"] = inflated_timestamps[:5]
    return report

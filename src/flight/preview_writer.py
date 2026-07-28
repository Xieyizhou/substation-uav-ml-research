"""Offline preview writer for A* routes.

This module is used by dry-run preview commands. It writes a machine-readable
`path_preview.json` and a visual `grid_path.png` showing raw obstacle
footprints, inflated planning cells, the A* grid path, simplified waypoints,
and optional return-home route. It does not connect to PX4 or MAVSDK.
"""

import json

import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

from src.planner.astar_grid import cell_to_local_waypoint


def list_cells(cells):
    """Convert `(x, y)` tuples into JSON-friendly two-item lists."""
    return [[x, y] for x, y in cells]


def cell_name_map_to_json(cell_name_map):
    """Convert a cell-to-name mapping into stable JSON entries."""
    return [
        {"cell": [cell[0], cell[1]], "names": names.split(",") if names else []}
        for cell, names in sorted(cell_name_map.items())
    ]


def save_preview(
    grid_path,
    simplified_path,
    waypoints,
    args,
    planner_config,
    preview_dir,
    planner_name,
    display_path_func,
    return_waypoints_func,
):
    """Write route preview JSON and PNG files for a planned A* route.

    Args:
        grid_path: Full A* cell-by-cell path.
        simplified_path: Reduced waypoint path after line simplification.
        waypoints: Local NED waypoints generated from the simplified path.
        args: Parsed CLI args, including return-home and diagonal settings.
        planner_config: Normalized map and obstacle metadata.
        preview_dir: Output directory for preview artifacts.
        planner_name: Planner identifier written to JSON.
        display_path_func: Helper for project-relative path display.
        return_waypoints_func: Helper that mirrors outbound waypoints for return-home.

    Side effects:
        Creates `preview_dir`, writes `path_preview.json`, writes
        `grid_path.png`, and prints the output paths.
    """
    preview_dir.mkdir(parents=True, exist_ok=True)
    return_grid_path = list(reversed(grid_path)) if args.return_home else []
    return_simplified_path = list(reversed(simplified_path)) if args.return_home else []
    return_waypoints = return_waypoints_func(waypoints) if args.return_home else []
    start_local = cell_to_local_waypoint(
        planner_config["start"],
        planner_config["resolution_m"],
        planner_config["altitude_m"],
    )
    goal_local = cell_to_local_waypoint(
        planner_config["goal"],
        planner_config["resolution_m"],
        planner_config["altitude_m"],
    )

    preview_json = {
        "planner_name": planner_name,
        "map_name": planner_config["map_name"],
        "target_id": planner_config.get("target_id"),
        "target_display_name": planner_config.get("target_display_name"),
        "gazebo_world_origin_m": planner_config.get(
            "gazebo_world_origin_m", [0.0, 0.0, 0.0]
        ),
        "obstacle_config": (
            str(display_path_func(planner_config["obstacle_config_path"]))
            if planner_config["obstacle_config_path"] is not None
            else None
        ),
        "grid_width": planner_config["width"],
        "grid_height": planner_config["height"],
        "start_cell": list(planner_config["start"]),
        "goal_cell": list(planner_config["goal"]),
        "grid_start": list(planner_config["start"]),
        "grid_goal": list(planner_config["goal"]),
        "start_local": start_local,
        "goal_local": goal_local,
        "allow_diagonal": args.allow_diagonal,
        "resolution_m": planner_config["resolution_m"],
        "altitude_m": planner_config["altitude_m"],
        "vertical_safety_margin_m": planner_config["vertical_safety_margin_m"],
        "horizontal_inflation_cells": planner_config["horizontal_inflation_cells"],
        "blocking_obstacle_names": planner_config["blocking_obstacle_names"],
        "nonblocking_obstacle_names": planner_config["nonblocking_obstacle_names"],
        "obstacles": list_cells(sorted(planner_config["obstacles"])),
        "raw_obstacle_cells": list_cells(sorted(planner_config["raw_obstacle_cells"])),
        "raw_blocking_cells": list_cells(sorted(planner_config["raw_blocking_cells"])),
        "inflated_blocking_cells": list_cells(sorted(planner_config["inflated_blocking_cells"])),
        "blocking_obstacle_cells": list_cells(sorted(planner_config["blocking_obstacle_cells"])),
        "inflated_obstacle_cells": list_cells(sorted(planner_config["inflated_obstacle_cells"])),
        "raw_obstacle_cell_to_name": cell_name_map_to_json(planner_config["raw_obstacle_cell_to_name"]),
        "inflated_obstacle_cell_to_name": cell_name_map_to_json(planner_config["inflated_obstacle_cell_to_name"]),
        "raw_obstacle_cell_count": planner_config["raw_obstacle_cell_count"],
        "raw_blocking_cell_count": planner_config["raw_blocking_cell_count"],
        "obstacle_cell_count": planner_config["obstacle_cell_count"],
        "inflated_obstacle_cell_count": planner_config["inflated_obstacle_cell_count"],
        "original_grid_path": list_cells(grid_path),
        "simplified_grid_path": list_cells(simplified_path),
        "generated_local_waypoints": waypoints,
        "grid_path": list_cells(grid_path),
        "simplified_path": list_cells(simplified_path),
        "waypoints": waypoints,
        "return_home_enabled": args.return_home,
        "return_home": args.return_home,
        "return_grid_path": list_cells(return_grid_path),
        "return_simplified_path": list_cells(return_simplified_path),
        "return_waypoints": return_waypoints,
        "coordinate_note": "grid x = east, grid y = north; cell centers convert to local NED waypoints",
        "validation_warnings": planner_config.get("validation_warnings", []),
    }
    json_path = preview_dir / "path_preview.json"
    json_path.write_text(json.dumps(preview_json, indent=2) + "\n")

    fig, ax = plt.subplots(figsize=(8, 8))
    ax.set_title(
        "A* Grid Path Preview\n"
        f"Map: {planner_config['map_name']} | "
        f"Target: {planner_config.get('target_display_name') or 'config default'} | "
        f"altitude={planner_config['altitude_m']} m | "
        f"vertical margin={planner_config['vertical_safety_margin_m']} m | "
        f"inflation={planner_config['horizontal_inflation_cells']} cell(s)\n"
        "A* uses inflated cells; Gazebo objects correspond to raw footprints."
    )
    ax.set_xlim(-0.5, planner_config["width"] - 0.5)
    ax.set_ylim(-0.5, planner_config["height"] - 0.5)
    ax.set_xticks(range(planner_config["width"]))
    ax.set_yticks(range(planner_config["height"]))
    ax.grid(True)
    ax.set_aspect("equal")
    ax.set_xlabel("Grid x / east")
    ax.set_ylabel("Grid y / north")

    first_raw_obstacle = next(iter(sorted(planner_config["raw_obstacle_cells"])), None)
    for obstacle in planner_config["raw_obstacle_cells"]:
        ax.add_patch(
            Rectangle(
                (obstacle[0] - 0.5, obstacle[1] - 0.5),
                1,
                1,
                facecolor="0.78",
                edgecolor="0.60",
                alpha=0.7,
                label="raw physical footprint" if obstacle == first_raw_obstacle else None,
            )
        )

    first_inflated_obstacle = next(iter(sorted(planner_config["inflated_blocking_cells"])), None)
    for obstacle in planner_config["inflated_blocking_cells"]:
        ax.add_patch(
            Rectangle(
                (obstacle[0] - 0.5, obstacle[1] - 0.5),
                1,
                1,
                facecolor="0.25",
                edgecolor="0.15",
                alpha=0.30,
                hatch="//",
                label="inflated planning obstacle" if obstacle == first_inflated_obstacle else None,
            )
        )

    path_x = [cell[0] for cell in grid_path]
    path_y = [cell[1] for cell in grid_path]
    simplified_x = [cell[0] for cell in simplified_path]
    simplified_y = [cell[1] for cell in simplified_path]

    ax.plot(path_x, path_y, "o-", color="tab:blue", label="outbound A* path", linewidth=1.5)
    ax.plot(
        simplified_x,
        simplified_y,
        "x--",
        color="tab:orange",
        label="simplified waypoints",
        markersize=8,
        linewidth=1.4,
    )

    if args.return_home:
        return_path_x = [cell[0] for cell in return_grid_path]
        return_path_y = [cell[1] for cell in return_grid_path]
        return_simplified_x = [cell[0] for cell in return_simplified_path]
        return_simplified_y = [cell[1] for cell in return_simplified_path]
        ax.plot(
            return_path_x,
            return_path_y,
            "o--",
            color="tab:orange",
            label="return path",
            linewidth=1.2,
        )
        ax.plot(
            return_simplified_x,
            return_simplified_y,
            "d:",
            color="tab:red",
            label="return waypoints",
            linewidth=1.2,
        )

    ax.scatter(*planner_config["start"], marker="o", s=140, color="limegreen", label="start", zorder=4)
    ax.scatter(*planner_config["goal"], marker="*", s=220, color="red", label="goal", zorder=4)

    for index, cell in enumerate(simplified_path, start=1):
        offset = (6, 6) if index % 2 else (6, -12)
        ax.annotate(f"WP{index:02d}", cell, textcoords="offset points", xytext=offset)

    ax.text(
        0.01,
        -0.08,
        "Coordinate note: grid x = east, grid y = north; local waypoint uses cell center (x+0.5, y+0.5).",
        transform=ax.transAxes,
        fontsize=9,
    )

    ax.legend()
    fig.tight_layout()
    image_path = preview_dir / "grid_path.png"
    fig.savefig(image_path, dpi=150)
    plt.close(fig)

    if args.return_home:
        print("Return path: reversed outbound path.")
    print("Coordinate note: grid x = east, grid y = north.")
    print(f"Preview image saved to {image_path}")
    print(f"Preview JSON saved to {json_path}")

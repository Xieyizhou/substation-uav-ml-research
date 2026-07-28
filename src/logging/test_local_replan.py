import json
import os
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.planner.astar_grid import astar
from src.planner.obstacle_config import (
    build_obstacle_map,
    get_resolution_altitude,
    get_start_goal,
    inflate_cells,
    load_obstacle_config,
)
from src.logging.output_registry import get_previews_dir


OUTPUT_DIR = get_previews_dir("replan_log_only") / "replan_preview"
OBSTACLE_CONFIG = PROJECT_ROOT / "config" / "substation_obstacles.json"
DYNAMIC_OBSTACLE_INFLATION_CELLS = 1

# Keep matplotlib cache files inside the project, matching the active analysis scripts.
MPLCONFIG_DIR = PROJECT_ROOT / "outputs" / ".matplotlib_cache"
MPLCONFIG_DIR.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", str(MPLCONFIG_DIR))


def cell_center(cell, resolution_m):
    x, y = cell
    return (x + 0.5) * resolution_m, (y + 0.5) * resolution_m


def choose_dynamic_blockage(path, existing_obstacles, width, height):
    middle_index = len(path) // 2
    candidate_indexes = range(
        max(1, middle_index - 3),
        min(len(path) - 1, middle_index + 4),
    )

    for index in candidate_indexes:
        cells = set(path[index : index + 2])
        inflated = inflate_cells(
            cells,
            width,
            height,
            DYNAMIC_OBSTACLE_INFLATION_CELLS,
        )
        if inflated - existing_obstacles:
            replan_index = max(0, index - 3)
            return index, cells, inflated, path[replan_index]

    raise RuntimeError("Could not choose a dynamic blockage on the original path")


def plot_cells(ax, cells, resolution_m, facecolor, edgecolor, label, alpha=0.8, hatch=None):
    from matplotlib.patches import Rectangle

    if not cells:
        return
    first = True
    for x, y in sorted(cells):
        ax.add_patch(
            Rectangle(
                (x * resolution_m, y * resolution_m),
                resolution_m,
                resolution_m,
                facecolor=facecolor,
                edgecolor=edgecolor,
                alpha=alpha,
                hatch=hatch,
                label=label if first else None,
            )
        )
        first = False


def plot_path(ax, path, resolution_m, color, label, linestyle="-", marker="o"):
    if not path:
        return
    points = [cell_center(cell, resolution_m) for cell in path]
    east = [point[0] for point in points]
    north = [point[1] for point in points]
    ax.plot(
        east,
        north,
        color=color,
        linewidth=2,
        linestyle=linestyle,
        marker=marker,
        markersize=3,
        label=label,
    )


def save_preview(
    output_path,
    width,
    height,
    resolution_m,
    static_obstacles,
    dynamic_blocked_cells,
    dynamic_inflated_cells,
    original_path,
    replanned_path,
    replan_start,
    goal,
):
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(8, 8))
    plot_cells(
        ax,
        static_obstacles,
        resolution_m,
        facecolor="0.75",
        edgecolor="0.55",
        label="static inflated obstacles",
        alpha=0.35,
        hatch="//",
    )
    plot_cells(
        ax,
        dynamic_inflated_cells,
        resolution_m,
        facecolor="tab:red",
        edgecolor="darkred",
        label="dynamic obstacle inflation",
        alpha=0.25,
        hatch="xx",
    )
    plot_cells(
        ax,
        dynamic_blocked_cells,
        resolution_m,
        facecolor="tab:red",
        edgecolor="darkred",
        label="dynamic blocked cells",
        alpha=0.75,
    )
    plot_path(
        ax,
        original_path,
        resolution_m,
        color="tab:blue",
        label="original A* path",
        linestyle="--",
        marker=".",
    )
    plot_path(
        ax,
        replanned_path,
        resolution_m,
        color="tab:green",
        label="replanned path",
        linestyle="-",
        marker="o",
    )

    replan_east, replan_north = cell_center(replan_start, resolution_m)
    goal_east, goal_north = cell_center(goal, resolution_m)
    ax.scatter(
        [replan_east],
        [replan_north],
        color="black",
        s=70,
        marker="s",
        label="replan start",
        zorder=5,
    )
    ax.scatter(
        [goal_east],
        [goal_north],
        color="gold",
        edgecolor="black",
        s=90,
        marker="*",
        label="goal",
        zorder=5,
    )

    ax.set_xlim(0, width * resolution_m)
    ax.set_ylim(0, height * resolution_m)
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlabel("East (m)")
    ax.set_ylabel("North (m)")
    ax.set_title("Dry-Run Local Replan Preview")
    ax.grid(True, linewidth=0.5, alpha=0.5)
    ax.legend(loc="best")
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)


def main():
    config = load_obstacle_config(OBSTACLE_CONFIG)
    width = int(config["width"])
    height = int(config["height"])
    start, goal = get_start_goal(config)
    resolution_m, altitude_m = get_resolution_altitude(config)
    obstacle_map = build_obstacle_map(
        config,
        flight_altitude_m=altitude_m,
        start_cell=start,
        goal_cell=goal,
    )
    static_obstacles = set(obstacle_map["inflated_blocking_cells"])

    original_path = astar(
        start=start,
        goal=goal,
        obstacles=static_obstacles,
        width=width,
        height=height,
    )
    _, dynamic_blocked_cells, dynamic_inflated_cells, replan_start = choose_dynamic_blockage(
        original_path,
        static_obstacles,
        width,
        height,
    )

    combined_obstacles = static_obstacles | dynamic_inflated_cells
    replanned_path = []
    replan_error = None
    try:
        replanned_path = astar(
            start=replan_start,
            goal=goal,
            obstacles=combined_obstacles,
            width=width,
            height=height,
        )
    except ValueError as error:
        replan_error = str(error)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    preview_path = OUTPUT_DIR / "local_replan_preview.png"
    json_path = OUTPUT_DIR / "local_replan_preview.json"
    save_preview(
        preview_path,
        width,
        height,
        resolution_m,
        static_obstacles,
        dynamic_blocked_cells,
        dynamic_inflated_cells,
        original_path,
        replanned_path,
        replan_start,
        goal,
    )

    summary = {
        "obstacle_config": str(OBSTACLE_CONFIG.relative_to(PROJECT_ROOT)),
        "original_path_length": len(original_path),
        "dynamic_blocked_cells": sorted(dynamic_blocked_cells),
        "dynamic_inflated_cells": sorted(dynamic_inflated_cells),
        "replan_start_cell": replan_start,
        "goal_cell": goal,
        "new_path_found": bool(replanned_path),
        "replanned_path_length": len(replanned_path),
        "preview_plot": str(preview_path.relative_to(PROJECT_ROOT)),
        "replan_error": replan_error,
    }
    json_path.write_text(json.dumps(summary, indent=2) + "\n")

    print("Local replan dry run")
    print(f"  original path length: {summary['original_path_length']}")
    print(f"  dynamic blocked cells: {summary['dynamic_blocked_cells']}")
    print(f"  replan start cell: {summary['replan_start_cell']}")
    print(f"  goal cell: {summary['goal_cell']}")
    print(f"  new path found: {str(summary['new_path_found']).lower()}")
    print(f"  replanned path length: {summary['replanned_path_length']}")
    if replan_error:
        print(f"  replan error: {replan_error}")
    print(f"  preview plot: {summary['preview_plot']}")
    print(f"  preview JSON: {json_path.relative_to(PROJECT_ROOT)}")


if __name__ == "__main__":
    main()

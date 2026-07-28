"""Target and collision diagnostic plots."""

import matplotlib.pyplot as plt
import pandas as pd

from src.logging.plot_trajectory import plot_obstacle_layers


def save_target_timeline(df, output_path, source_log):
    required_columns = ["elapsed_s", "target_name"]
    if not all(column in df.columns for column in required_columns):
        print("Skipping target timeline: target_name or elapsed_s is missing")
        return None

    columns = ["elapsed_s", "target_name"]
    if "route_direction" in df.columns:
        columns.append("route_direction")
    plot_df = df[columns].dropna(subset=["elapsed_s"])
    if "route_direction" not in plot_df.columns:
        plot_df["route_direction"] = "none"
    plot_df = plot_df[plot_df["target_name"].astype(str) != ""]
    if plot_df.empty:
        print("Skipping target timeline: no target data")
        return None

    ordered_targets = []
    for name in plot_df["target_name"]:
        if name not in ordered_targets:
            ordered_targets.append(name)
    target_to_index = {target: index for index, target in enumerate(ordered_targets, start=1)}
    plot_df = plot_df.copy()
    plot_df["target_index"] = plot_df["target_name"].map(target_to_index)

    plt.figure(figsize=(10, 5))
    for route_direction, group in plot_df.groupby("route_direction", dropna=False):
        label = route_direction if route_direction else "none"
        plt.scatter(group["elapsed_s"], group["target_index"], s=12, label=label)
    plt.yticks(list(target_to_index.values()), list(target_to_index.keys()))
    plt.title(f"A* Target Timeline\nSource: {source_log}")
    plt.xlabel("Elapsed time (s)")
    plt.ylabel("Target waypoint")
    plt.grid(True, axis="x")
    plt.legend(title="route_direction")
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()
    print(f"Saved plot: {output_path}")
    return output_path

def save_collision_zoom_plot(df, output_path, obstacle_map, resolution_m, collision_report):
    rows = collision_report.get("collision_rows", [])
    position_df = df[["local_east_m", "local_north_m"]].dropna()
    if rows:
        collision_df = pd.DataFrame(rows)
        east_min = collision_df["local_east_m"].min() - 3
        east_max = collision_df["local_east_m"].max() + 3
        north_min = collision_df["local_north_m"].min() - 3
        north_max = collision_df["local_north_m"].max() + 3
    elif not position_df.empty:
        east_min = position_df["local_east_m"].min() - 1
        east_max = position_df["local_east_m"].max() + 1
        north_min = position_df["local_north_m"].min() - 1
        north_max = position_df["local_north_m"].max() + 1
    else:
        return None

    fig, ax = plt.subplots(figsize=(7, 7))
    plot_obstacle_layers(ax, obstacle_map, resolution_m)
    ax.plot(
        position_df["local_east_m"],
        position_df["local_north_m"],
        color="tab:blue",
        linewidth=1.2,
        label="actual trajectory",
    )
    if rows:
        raw_df = collision_df[collision_df["entry_type"] == "raw_physical"]
        inflated_df = collision_df[collision_df["entry_type"] == "inflated_buffer"]
        if not raw_df.empty:
            ax.scatter(
                raw_df["local_east_m"],
                raw_df["local_north_m"],
                color="red",
                s=32,
                marker="x",
                label="raw footprint entry",
                zorder=4,
            )
        if not inflated_df.empty:
            ax.scatter(
                inflated_df["local_east_m"],
                inflated_df["local_north_m"],
                facecolors="none",
                edgecolors="black",
                s=36,
                marker="o",
                label="inflated buffer entry",
                zorder=4,
            )
    else:
        ax.text(
            0.5,
            0.95,
            "No hard obstacle cell entries detected",
            transform=ax.transAxes,
            ha="center",
            va="top",
            bbox={"boxstyle": "round,pad=0.25", "fc": "white", "alpha": 0.8},
        )
    ax.set_xlim(east_min, east_max)
    ax.set_ylim(north_min, north_max)
    ax.set_title("Obstacle Collision Zoom")
    ax.set_xlabel("East (m)")
    ax.set_ylabel("North (m)")
    ax.set_aspect("equal", adjustable="box")
    ax.grid(True)
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)
    print(f"Saved plot: {output_path}")
    return output_path

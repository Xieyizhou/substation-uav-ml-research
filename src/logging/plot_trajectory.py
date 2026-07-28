"""Trajectory and obstacle-map plotting helpers."""

import math

import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.patches import Rectangle

from src.logging.log_io import display_path
from src.logging.metrics import (
    detected_obstacle_mask,
    has_perception_columns,
    risk_level_series,
)


PLOT_SPLIT_DISTANCE_M = 3.0


def target_sequence(df, route_direction=None):
    required_columns = ["target_name", "target_north_m", "target_east_m", "target_down_m"]
    if not all(column in df.columns for column in required_columns):
        return []

    target_df = df[required_columns + [col for col in ["route_direction"] if col in df.columns]].dropna(
        subset=["target_north_m", "target_east_m"]
    )
    if route_direction is not None and "route_direction" in target_df.columns:
        target_df = target_df[target_df["route_direction"] == route_direction]

    targets = []
    seen = set()
    for _, row in target_df.iterrows():
        name = str(row["target_name"])
        if not name or name == "nan":
            continue
        key = (route_direction, name)
        if key in seen:
            continue
        targets.append(
            {
                "name": name,
                "north_m": float(row["target_north_m"]),
                "east_m": float(row["target_east_m"]),
                "down_m": float(row["target_down_m"]),
            }
        )
        seen.add(key)
    return targets


def plot_obstacle_cells(ax, obstacles, resolution_m, label, facecolor, alpha, hatch=None):
    if not obstacles:
        return
    first = True
    for x, y in obstacles:
        ax.add_patch(
            Rectangle(
                (x * resolution_m, y * resolution_m),
                resolution_m,
                resolution_m,
                facecolor=facecolor,
                edgecolor="0.25",
                alpha=alpha,
                hatch=hatch,
                label=label if first else None,
            )
        )
        first = False


def plot_obstacle_layers(ax, obstacle_map, resolution_m):
    if not obstacle_map or not resolution_m:
        return
    plot_obstacle_cells(
        ax,
        obstacle_map.get("raw_obstacle_cells", set()),
        resolution_m,
        "raw physical footprint",
        "0.78",
        0.60,
    )
    plot_obstacle_cells(
        ax,
        obstacle_map.get("inflated_blocking_cells", set()),
        resolution_m,
        "inflated planning obstacle",
        "0.25",
        0.25,
        hatch="//",
    )


def trajectory_segments(df):
    required_columns = ["elapsed_s", "local_north_m", "local_east_m"]
    if not all(column in df.columns for column in required_columns):
        return []

    columns = required_columns + [
        column
        for column in ["phase", "route_direction", "target_name"]
        if column in df.columns
    ]
    plot_df = df[columns].dropna(subset=required_columns).sort_values("elapsed_s")
    if plot_df.empty:
        return []

    segments = []
    current_rows = []
    previous = None
    for _, row in plot_df.iterrows():
        split = False
        if previous is not None:
            dx = float(row["local_east_m"]) - float(previous["local_east_m"])
            dy = float(row["local_north_m"]) - float(previous["local_north_m"])
            jump = math.hypot(dx, dy)
            if jump > PLOT_SPLIT_DISTANCE_M:
                split = True
            if "phase" in plot_df.columns and row["phase"] != previous["phase"]:
                split = True
            if "route_direction" in plot_df.columns and row["route_direction"] != previous["route_direction"]:
                split = True
            if (
                "target_name" in plot_df.columns
                and row["target_name"] != previous["target_name"]
                and jump > PLOT_SPLIT_DISTANCE_M
            ):
                split = True

        if split and current_rows:
            segments.append(pd.DataFrame(current_rows))
            current_rows = []
        current_rows.append(row.to_dict())
        previous = row

    if current_rows:
        segments.append(pd.DataFrame(current_rows))
    return segments


def annotate_waypoints(ax, targets):
    offsets = [(6, 8), (8, -12), (-22, 8), (-24, -12), (10, 16), (-32, 0)]
    for index, target in enumerate(targets):
        offset = offsets[index % len(offsets)]
        ax.annotate(
            target["name"],
            (target["east_m"], target["north_m"]),
            textcoords="offset points",
            xytext=offset,
            fontsize=8,
            bbox={"boxstyle": "round,pad=0.15", "fc": "white", "alpha": 0.7, "lw": 0},
        )


def save_trajectory_plot(
    df,
    output_path,
    log_path,
    map_name,
    obstacle_map,
    resolution_m,
    collision_report=None,
    title="A* Local 2D Trajectory",
    show_obstacles=False,
    show_collision_points=False,
    show_perception_points=False,
):
    required_columns = ["local_north_m", "local_east_m"]
    if not all(column in df.columns for column in required_columns):
        print("Skipping trajectory plot: local position columns are missing")
        return None

    valid_df = df[required_columns].dropna()
    if valid_df.empty:
        print("Skipping trajectory plot: no valid local position data")
        return None

    fig, ax = plt.subplots(figsize=(8, 8))
    if show_obstacles:
        plot_obstacle_layers(ax, obstacle_map, resolution_m)

    colors = {
        "outbound": "tab:blue",
        "return": "tab:purple",
        "none": "0.45",
        "": "0.45",
    }
    labels_seen = set()
    for segment in trajectory_segments(df):
        route_direction = str(segment["route_direction"].iloc[0]) if "route_direction" in segment.columns else "none"
        phase = str(segment["phase"].iloc[0]) if "phase" in segment.columns else ""
        if route_direction == "outbound":
            label = "actual outbound path"
        elif route_direction == "return":
            label = "actual return path"
        elif phase in {"goal_hover", "start_hover", "landing", "landed"}:
            label = f"actual {phase}"
        else:
            label = "actual other/hover"
        ax.plot(
            segment["local_east_m"],
            segment["local_north_m"],
            marker="o",
            markersize=2,
            linewidth=1.4 if route_direction in {"outbound", "return"} else 0.9,
            color=colors.get(route_direction, "0.45"),
            alpha=0.9 if route_direction in {"outbound", "return"} else 0.55,
            label=label if label not in labels_seen else None,
        )
        labels_seen.add(label)

    outbound_targets = target_sequence(df, route_direction="outbound")
    if not outbound_targets:
        outbound_targets = target_sequence(df)
    if outbound_targets:
        target_east = [target["east_m"] for target in outbound_targets]
        target_north = [target["north_m"] for target in outbound_targets]
        ax.plot(
            target_east,
            target_north,
            "x--",
            color="tab:orange",
            markersize=8,
            linewidth=1.5,
            label="planned A* waypoints",
        )
        annotate_waypoints(ax, outbound_targets)

    if show_collision_points and collision_report:
        raw_rows = collision_report.get("raw_collision_rows", [])
        inflated_rows = collision_report.get("inflated_buffer_entry_rows", [])
        if raw_rows:
            raw_df = pd.DataFrame(raw_rows)
            ax.scatter(
                raw_df["local_east_m"],
                raw_df["local_north_m"],
                color="red",
                s=30,
                marker="x",
                label="raw footprint entry",
                zorder=5,
            )
        if inflated_rows:
            inflated_df = pd.DataFrame(inflated_rows)
            ax.scatter(
                inflated_df["local_east_m"],
                inflated_df["local_north_m"],
                facecolors="none",
                edgecolors="black",
                s=36,
                marker="o",
                label="inflated buffer entry",
                zorder=5,
            )

    if show_perception_points and has_perception_columns(df):
        perception_df = df[
            detected_obstacle_mask(df)
            & df["local_east_m"].notna()
            & df["local_north_m"].notna()
        ].copy()
        if not perception_df.empty:
            perception_df["risk_level"] = risk_level_series(perception_df)
        detected_df = perception_df[perception_df["risk_level"] == "detected"] if not perception_df.empty else pd.DataFrame()
        warning_df = perception_df[perception_df["risk_level"] == "warning"] if not perception_df.empty else pd.DataFrame()
        danger_df = perception_df[perception_df["risk_level"] == "danger"] if not perception_df.empty else pd.DataFrame()
        if not detected_df.empty:
            ax.scatter(
                detected_df["local_east_m"],
                detected_df["local_north_m"],
                color="tab:blue",
                s=10,
                marker="o",
                alpha=0.25,
                label="perception detected sample",
                zorder=5,
            )
        if not warning_df.empty:
            ax.scatter(
                warning_df["local_east_m"],
                warning_df["local_north_m"],
                color="orange",
                s=28,
                marker="o",
                alpha=0.8,
                label="perception warning sample",
                zorder=6,
            )
        if not danger_df.empty:
            ax.scatter(
                danger_df["local_east_m"],
                danger_df["local_north_m"],
                color="red",
                s=34,
                marker="o",
                alpha=0.9,
                label="perception danger sample",
                zorder=6,
            )

    first_position = df[["local_east_m", "local_north_m"]].dropna().iloc[0]
    final_position = df[["local_east_m", "local_north_m"]].dropna().iloc[-1]
    ax.scatter(
        first_position["local_east_m"],
        first_position["local_north_m"],
        marker="o",
        s=120,
        color="limegreen",
        edgecolor="black",
        label="start",
        zorder=4,
    )
    ax.scatter(
        final_position["local_east_m"],
        final_position["local_north_m"],
        marker="*",
        s=180,
        color="red",
        edgecolor="black",
        label="final/end",
        zorder=4,
    )

    map_text = map_name or "N/A"
    ax.set_title(
        f"{title}\n"
        f"Source: {display_path(log_path)} | Map: {map_text}"
    )
    ax.set_xlabel("East (m)")
    ax.set_ylabel("North (m)")
    ax.set_aspect("equal", adjustable="box")
    ax.grid(True)
    ax.legend(loc="best", fontsize=8)
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)
    print(f"Saved plot: {output_path}")
    return output_path

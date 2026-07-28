"""Time-series plots for offline A* analysis reports."""

import matplotlib.pyplot as plt

from src.logging.metrics import (
    DEFAULT_DANGER_DISTANCE_M,
    DEFAULT_WARNING_DISTANCE_M,
    RISK_LEVEL_TO_VALUE,
    first_active_value,
    perception_enabled_mask,
    risk_level_series,
)


WAYPOINT_REACHED_THRESHOLD_M = 0.4


def save_line_plot(df, columns, title, ylabel, output_path, source_log=None):
    available_columns = [column for column in columns if column in df.columns]
    if not available_columns:
        print(f"Skipping {title}: missing columns {columns}")
        return None

    plot_df = df[["elapsed_s", *available_columns]].dropna(
        subset=available_columns,
        how="all",
    )
    if plot_df.empty:
        print(f"Skipping {title}: no non-empty data points")
        return None

    plt.figure(figsize=(10, 5))
    for column in available_columns:
        plt.plot(plot_df["elapsed_s"], plot_df[column], label=column)
    subtitle = f"\nSource: {source_log}" if source_log else ""
    plt.title(f"{title}{subtitle}")
    plt.xlabel("Elapsed time (s)")
    plt.ylabel(ylabel)
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()
    print(f"Saved plot: {output_path}")
    return output_path

def save_perception_timeline(df, output_path, source_log):
    required_columns = {"elapsed_s", "perception_enabled", "nearest_obstacle_distance_m"}
    if not required_columns.issubset(df.columns):
        print("Skipping perception timeline: perception distance columns are missing")
        return None

    plot_df = df[perception_enabled_mask(df)].dropna(
        subset=["elapsed_s", "nearest_obstacle_distance_m"]
    )
    if plot_df.empty:
        print("Skipping perception timeline: no active detection distance samples")
        return None

    plt.figure(figsize=(10, 5))
    plt.plot(
        plot_df["elapsed_s"],
        plot_df["nearest_obstacle_distance_m"],
        marker="o",
        markersize=3,
        linewidth=1.2,
        label="nearest detected obstacle",
    )
    warning_distance = first_active_value(df, "warning_distance_m")
    danger_distance = first_active_value(df, "danger_distance_m")
    warning_distance = DEFAULT_WARNING_DISTANCE_M if warning_distance is None else warning_distance
    danger_distance = DEFAULT_DANGER_DISTANCE_M if danger_distance is None else danger_distance
    if warning_distance is not None:
        plt.axhline(
            float(warning_distance),
            color="orange",
            linestyle="--",
            linewidth=1.1,
            label="warning threshold",
        )
    if danger_distance is not None:
        plt.axhline(
            float(danger_distance),
            color="red",
            linestyle="--",
            linewidth=1.1,
            label="danger threshold",
        )
    plt.title(f"Perception Nearest Obstacle Distance\nSource: {source_log}")
    plt.xlabel("Elapsed time (s)")
    plt.ylabel("Nearest obstacle distance (m)")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()
    print(f"Saved plot: {output_path}")
    return output_path


def save_perception_risk_timeline(df, output_path, source_log):
    required_columns = {"elapsed_s", "perception_enabled"}
    if not required_columns.issubset(df.columns):
        print("Skipping perception risk timeline: perception columns are missing")
        return None

    plot_df = df[perception_enabled_mask(df)].copy()
    if plot_df.empty:
        print("Skipping perception risk timeline: no active perception samples")
        return None

    plot_df["risk_value"] = risk_level_series(plot_df).map(RISK_LEVEL_TO_VALUE)
    plot_df = plot_df.dropna(subset=["elapsed_s", "risk_value"])
    if plot_df.empty:
        print("Skipping perception risk timeline: no risk-level samples")
        return None

    risk_action = first_active_value(df, "risk_action") or "log_only"
    plt.figure(figsize=(10, 5))
    plt.step(
        plot_df["elapsed_s"],
        plot_df["risk_value"],
        where="post",
        linewidth=1.4,
        label="perception risk level",
    )
    plt.yticks(
        [0, 1, 2, 3],
        ["clear = 0", "detected = 1", "warning = 2", "danger = 3"],
    )
    plt.ylim(-0.25, 3.25)
    plt.title(
        "Perception Risk Level Over Time"
        f"\nSource: {source_log} | risk_action: {risk_action}"
    )
    plt.xlabel("Elapsed time (s)")
    plt.ylabel("Risk level (clear=0, detected=1, warning=2, danger=3)")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()
    print(f"Saved plot: {output_path}")
    return output_path


def save_detection_count_plot(df, output_path, source_log):
    required_columns = {"elapsed_s", "perception_enabled", "detected_obstacle_count"}
    if not required_columns.issubset(df.columns):
        print("Skipping detection count plot: perception count columns are missing")
        return None

    plot_df = df[perception_enabled_mask(df)].dropna(
        subset=["elapsed_s", "detected_obstacle_count"]
    )
    if plot_df.empty:
        print("Skipping detection count plot: no active perception count samples")
        return None

    plt.figure(figsize=(10, 5))
    plt.plot(
        plot_df["elapsed_s"],
        plot_df["detected_obstacle_count"],
        marker=".",
        linewidth=1.1,
        label="detected obstacle count",
    )
    plt.title(f"Perception Detection Count Over Time\nSource: {source_log}")
    plt.xlabel("Elapsed time (s)")
    plt.ylabel("Detected obstacle count")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()
    print(f"Saved plot: {output_path}")
    return output_path


def save_error_plot(df, output_path, source_log):
    if "horizontal_error_m" not in df.columns:
        print("Skipping error plot: horizontal_error_m is missing")
        return None

    plot_df = df[["elapsed_s", "horizontal_error_m", "target_name"]].dropna(
        subset=["elapsed_s", "horizontal_error_m"]
    )
    if plot_df.empty:
        print("Skipping error plot: no horizontal error data")
        return None

    plt.figure(figsize=(10, 5))
    plt.plot(plot_df["elapsed_s"], plot_df["horizontal_error_m"], label="horizontal_error_m")
    plt.axhline(
        WAYPOINT_REACHED_THRESHOLD_M,
        color="tab:green",
        linestyle="--",
        linewidth=1,
        label=f"reached threshold {WAYPOINT_REACHED_THRESHOLD_M:.1f} m",
    )

    if "target_name" in plot_df.columns:
        last_target = None
        for _, row in plot_df.iterrows():
            target = str(row["target_name"])
            if target and target != last_target:
                plt.axvline(row["elapsed_s"], color="0.75", linestyle=":", linewidth=0.8)
                last_target = target

    plt.title(f"Horizontal A* Waypoint Error Over Time\nSource: {source_log}")
    plt.xlabel("Elapsed time (s)")
    plt.ylabel("Horizontal error (m)")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()
    print(f"Saved plot: {output_path}")
    return output_path

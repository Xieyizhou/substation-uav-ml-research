"""Compatibility exports for offline A* analysis plotting helpers."""

from src.logging.plot_diagnostics import (
    save_collision_zoom_plot,
    save_target_timeline,
)
from src.logging.plot_timeseries import (
    WAYPOINT_REACHED_THRESHOLD_M,
    save_detection_count_plot,
    save_error_plot,
    save_line_plot,
    save_perception_risk_timeline,
    save_perception_timeline,
)
from src.logging.plot_trajectory import (
    PLOT_SPLIT_DISTANCE_M,
    annotate_waypoints,
    plot_obstacle_cells,
    plot_obstacle_layers,
    save_trajectory_plot,
    target_sequence,
    trajectory_segments,
)


__all__ = [
    "PLOT_SPLIT_DISTANCE_M",
    "WAYPOINT_REACHED_THRESHOLD_M",
    "annotate_waypoints",
    "plot_obstacle_cells",
    "plot_obstacle_layers",
    "save_collision_zoom_plot",
    "save_detection_count_plot",
    "save_error_plot",
    "save_line_plot",
    "save_perception_risk_timeline",
    "save_perception_timeline",
    "save_target_timeline",
    "save_trajectory_plot",
    "target_sequence",
    "trajectory_segments",
]

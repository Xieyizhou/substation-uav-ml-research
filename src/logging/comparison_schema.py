"""Shared schemas for cross-stage experiment comparisons."""

from src.logging.evaluation_schema import EVALUATION_COLUMNS
from src.logging.output_registry import RUN_STAGES


MARKERS = [
    "KEEP_BASELINE__astar_only_perception_disabled.txt",
    "KEEP_COMPARISON__perception_slow_down.txt",
    "KEEP_CONTROL__log_only_local_replan_no_route_replacement.txt",
    "KEEP_LANDMARK__active_local_replan.txt",
]

LANDMARK_COMPARISON_NAME = "landmark"
AGGREGATE_COMPARISON_NAME = "aggregate"
COMPARISON_COLUMNS = [*EVALUATION_COLUMNS, "marker", "run_dir"]
REQUIRED_STAGES = RUN_STAGES
STATUS_FILENAME = "comparison_status.md"
AGGREGATE_COLUMNS = [
    "stage",
    "stage_folder",
    "run_count",
    "completed_count",
    "pass_count",
    "mean_total_flight_time_s",
    "std_total_flight_time_s",
    "min_total_flight_time_s",
    "max_total_flight_time_s",
    "mean_planned_path_length_m",
    "mean_actual_traveled_distance_m",
    "mean_minimum_distance_to_obstacle_m",
    "total_safety_buffer_violation_count",
    "mean_perception_risk_detection_count",
    "mean_slow_down_event_count",
    "mean_local_replan_attempt_count",
    "mean_successful_local_replan_count",
    "mean_active_route_replacement_count",
]
INCLUDED_RUN_COLUMNS = [*EVALUATION_COLUMNS, "run_dir"]

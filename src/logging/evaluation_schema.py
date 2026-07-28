"""Shared formal evaluation table schema for experiment summaries and comparisons."""

EVALUATION_COLUMNS = [
    "run_id",
    "stage",
    "stage_folder",
    "experiment_type",
    "completed_or_failed",
    "total_flight_time_s",
    "planned_path_length_m",
    "actual_traveled_distance_m",
    "minimum_distance_to_obstacle_m",
    "safety_buffer_violation_count",
    "perception_risk_detection_count",
    "slow_down_event_count",
    "local_replan_attempt_count",
    "successful_local_replan_count",
    "active_route_replacement_count",
    "final_status",
    "notes",
]

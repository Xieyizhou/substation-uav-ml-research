"""Execute the bounded capability challenge flight tier."""

from src.study.closed_loop_worker import execute_flight_tier


def execute_challenge(registry_path, study_id, results_dir, **options):
    return execute_flight_tier(
        registry_path, study_id, results_dir, tier="challenge", **options
    )

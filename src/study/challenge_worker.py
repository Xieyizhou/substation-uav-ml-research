"""Execute the bounded capability challenge flight tier."""

from src.study.closed_loop_worker import execute_flight_tier
from src.study.challenge_receipt import materialize_challenge_receipt


def execute_challenge(registry_path, study_id, results_dir, **options):
    result = execute_flight_tier(
        registry_path, study_id, results_dir, tier="challenge", **options
    )
    receipt = materialize_challenge_receipt(registry_path, study_id, results_dir)
    return {**result, "challenge_receipt": receipt}

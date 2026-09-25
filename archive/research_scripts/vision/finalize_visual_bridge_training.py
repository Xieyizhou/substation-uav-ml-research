"""Close the visual-bridge A/B/C/D development grid without opening the sealed test."""
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from src.ml.artifacts import file_sha256, object_sha256, write_json

OUT = ROOT / "data/research/ml_training_recovery_v1/visual-bridge-training-v1"
SEEDS = (7, 17, 27)


def finalize():
    protocol_path = OUT / "protocol.json"
    progress_path = OUT / "training-progress.json"
    evaluation_path = OUT / "development-evaluation.json"
    protocol = json.loads(protocol_path.read_text())
    progress = json.loads(progress_path.read_text())
    evaluation = json.loads(evaluation_path.read_text())

    if protocol["status"] != "frozen_before_training":
        raise ValueError("Training protocol was not frozen before execution")
    if progress["status"] != "complete_pending_evaluation":
        raise ValueError("Training grid is incomplete")
    if evaluation["status"] != "development_complete_no_candidate" or evaluation["selected_arm"] is not None:
        raise ValueError("Development evaluation did not reject all candidates")
    if any(row["passed"] for row in evaluation["policy_results"].values()):
        raise ValueError("Evaluation status conflicts with the frozen policy")

    cells = []
    for arm in "ABCD":
        for seed in SEEDS:
            path = OUT / f"arm-{arm}-seed-{seed}/completion.json"
            row = json.loads(path.read_text())
            if (
                row["status"] != "complete"
                or row["protocol_identity"] != protocol["identity"]
                or row["observed_draws"] != 600
                or row["optimizer_steps"] != 100
                or not row["exposure_verified"]
            ):
                raise ValueError(f"Training cell is incomplete or inconsistent: {arm}/{seed}")
            weights = Path(row["weights"])
            if file_sha256(weights) != row["weights_sha256"]:
                raise ValueError(f"Training weight hash changed: {arm}/{seed}")
            cells.append(
                {
                    "arm": arm,
                    "seed": seed,
                    "completion_path": str(path),
                    "completion_sha256": file_sha256(path),
                    "weights_path": str(weights),
                    "weights_sha256": row["weights_sha256"],
                    "observed_draws": row["observed_draws"],
                    "optimizer_steps": row["optimizer_steps"],
                    "exposure_verified": True,
                }
            )

    result = {
        "schema_version": 1,
        "status": "development_complete_no_candidate",
        "decision": "No arm satisfies every pre-frozen development acceptance rule; no arm or seed is selected.",
        "candidate_selected": False,
        "selected_arm": None,
        "training_cells": cells,
        "protocol_identity": protocol["identity"],
        "protocol_sha256": file_sha256(protocol_path),
        "training_progress_identity": progress["identity"],
        "training_progress_sha256": file_sha256(progress_path),
        "development_evaluation_identity": evaluation["identity"],
        "development_evaluation_sha256": file_sha256(evaluation_path),
        "policy_results": evaluation["policy_results"],
        "closest_arm": "D",
        "closest_arm_summary": {
            "original_planned_instance_hit_rate_mean": evaluation["aggregate"]["D"]["original"]["planned_instance_hit_rate"]["mean"],
            "material_planned_instance_hit_rate_mean": evaluation["aggregate"]["D"]["material"]["planned_instance_hit_rate"]["mean"],
            "background_planned_instance_hit_rate_mean": evaluation["aggregate"]["D"]["background"]["planned_instance_hit_rate"]["mean"],
            "lighting_planned_instance_hit_rate_mean": evaluation["aggregate"]["D"]["lighting"]["planned_instance_hit_rate"]["mean"],
            "no_target_frame_false_positive_rate_mean": evaluation["aggregate"]["D"]["no_target"]["frame_false_positive_rate"]["mean"],
        },
        "next_recommendation": "Keep the unseen scene sealed. Diagnose transformer-like false positives in the fixed no-target regression and add source-matched hard negatives before freezing a new development composition.",
        "unseen_scene_status": "sealed_not_evaluated",
        "protected_label_accessed": False,
        "threshold_tuned": False,
        "training_admitted": False,
        "promotable": False,
    }
    result["identity"] = object_sha256(result)
    write_json(OUT / "completion.json", result)
    return result


if __name__ == "__main__":
    print(json.dumps(finalize(), indent=2))

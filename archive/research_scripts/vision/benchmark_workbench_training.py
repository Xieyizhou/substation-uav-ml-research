"""Compare completed paired training runs without changing either model."""

import argparse
import json
from pathlib import Path

from src.ml.artifacts import file_sha256, object_sha256, write_json
from src.sandbox.workbench_models import WorkbenchExperimentRecipe
from src.sandbox.workbench_run_guard import verified_completed_receipt


def read_run(root):
    root = Path(root)
    record = json.loads((root / "recipe.json").read_text())
    recipe = WorkbenchExperimentRecipe.from_record(record)
    receipt = verified_completed_receipt(root, recipe)
    if not receipt or "training_efficiency_sha256" not in receipt:
        raise ValueError("comparison requires a verified run with training telemetry")
    replay = json.loads((root / "replay.json").read_text())
    replay_identity = replay.pop("replay_identity_sha256")
    if object_sha256(replay) != replay_identity:
        raise ValueError("replay contents changed")
    view = json.loads((root / "view.json").read_text())
    members = json.loads((root / "view/membership.json").read_text())
    if object_sha256(members) != view["membership_sha256"]:
        raise ValueError("view membership changed")
    efficiency = json.loads((root / "training_efficiency.json").read_text())
    if efficiency["recipe_identity_sha256"] != recipe.recipe_identity_sha256 or efficiency["resumed"]:
        raise ValueError("comparison requires fresh training with matching telemetry")
    return {"run_root": str(root.resolve()), "recipe": record, "view": view,
            "receipt_identity_sha256": receipt["receipt_identity_sha256"],
            "training_efficiency": efficiency, "metrics": replay["metrics"],
            "runtime_identity_sha256": replay["runtime_identity_sha256"],
            "replay_sha256": file_sha256(root / "replay.json")}


def compare(baseline, candidate, output):
    baseline, candidate = read_run(baseline), read_run(candidate)
    for field in ("dataset_identity_sha256", "pretrained_weights_sha256", "seed"):
        if baseline["recipe"][field] != candidate["recipe"][field]:
            raise ValueError(f"paired experiment {field} differs")
    parameters = [{k: v for k, v in run["recipe"]["parameters"].items() if k != "freeze"}
                  for run in (baseline, candidate)]
    if (parameters[0] != parameters[1]
            or baseline["view"]["membership_sha256"] != candidate["view"]["membership_sha256"]
            or baseline["runtime_identity_sha256"] != candidate["runtime_identity_sha256"]):
        raise ValueError("paired settings, membership, or runtime differ")
    before, after = baseline["training_efficiency"], candidate["training_efficiency"]
    reductions = {}
    for field in ("elapsed_seconds", "max_resumable_checkpoint_bytes", "process_peak_rss_bytes", "final_best_bytes"):
        if before[field] <= 0 or after[field] <= 0:
            raise ValueError(f"missing positive {field}")
        reductions[field] = 1 - after[field] / before[field]
    result = {"baseline": baseline, "candidate": candidate, "reduction_fraction": reductions,
              "macro_f1_delta": candidate["metrics"]["macro_f1"] - baseline["metrics"]["macro_f1"],
              "scope": "One training run per strategy with matching settings and development membership; thresholds independently selected on validation. Not an independent retention or promotion gate.",
              "promoted": False}
    result["comparison_identity_sha256"] = object_sha256(result)
    write_json(output, result)
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline", type=Path, required=True)
    parser.add_argument("--candidate", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = compare(args.baseline, args.candidate, args.output)
    print(json.dumps({key: result[key] for key in ("reduction_fraction", "macro_f1_delta", "scope")}, indent=2))

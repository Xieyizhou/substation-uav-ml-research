"""Run the frozen 12-cell bridge grid sequentially with resumable receipts."""
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from src.ml.artifacts import object_sha256, write_json
from scripts.vision.prepare_visual_bridge_training import OUT, prepare
from scripts.vision.run_visual_bridge_training import train


def main():
    protocol = prepare(); completed = []
    for arm in "ABCD":
        for seed in (7, 17, 27):
            print(f"TRAIN {arm} seed={seed}", flush=True)
            result = train(arm, seed)
            completed.append({"arm": arm, "seed": seed, "completion_identity": result["identity"],
                              "weights_sha256": result["weights_sha256"], "exposure_verified": result["exposure_verified"]})
            progress = {"status": "running" if len(completed) < 12 else "complete_pending_evaluation",
                        "protocol_identity": protocol["identity"], "completed_cells": completed,
                        "training_admitted": False, "promotable": False,
                        "unseen_scene_status": "sealed_not_evaluated"}
            progress["identity"] = object_sha256(progress)
            write_json(OUT / "training-progress.json", progress)
    print(json.dumps(progress, indent=2))


if __name__ == "__main__":
    main()

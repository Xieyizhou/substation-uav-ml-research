"""Freeze A/B/C/D bridge-supplement memberships and deterministic schedules."""
import json
import random
import sys
from collections import Counter
from pathlib import Path

from PIL import Image

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from src.ml.artifacts import file_sha256, object_sha256, write_json

BASE = ROOT / "data/research/ml_training_recovery_v1"
OUT = BASE / "visual-bridge-training-v1"
ADMISSION = BASE / "visual-bridge-supplement-v2/development-admission.json"
PRIOR_PROTOCOL = BASE / "visual-augmentation-training-v2/protocol.json"
PAIRED_EVAL = BASE / "paired-visual-factors-v1/evaluation.json"
NEGATIVE_EVAL = BASE / "hard-negative-isolated-v2/semantic-review.json"
WEIGHTS = ROOT / "models/equipment/visual-yolo11n-baseline-v2.11-candidate-package-v1/weights/best.pt"
NAMES = ["transformer", "switchgear", "capacitor_bank", "reactor"]
SEEDS = (7, 17, 27)
EPOCHS, SLOTS, BATCH = 10, 60, 6


def yolo_lines(truth):
    lines = []
    for obj in truth["objects"]:
        x1, y1, x2, y2 = obj["bbox_xyxy"]
        if obj["class_name"] not in NAMES or not (0 <= x1 < x2 <= 1920 and 0 <= y1 < y2 <= 1080):
            raise ValueError("Invalid full_2d truth object")
        lines.append(f"{NAMES.index(obj['class_name'])} {(x1+x2)/3840:.10f} {(y1+y2)/2160:.10f} {(x2-x1)/1920:.10f} {(y2-y1)/1080:.10f}")
    return "\n".join(lines) + ("\n" if lines else "")


def schedule(member_ids, seed):
    rng = random.Random(seed); draws = []
    while len(draws) < EPOCHS * SLOTS:
        cycle = list(member_ids); rng.shuffle(cycle); draws.extend(cycle)
    draws = draws[:EPOCHS * SLOTS]
    return [draws[index*SLOTS:(index+1)*SLOTS] for index in range(EPOCHS)]


def prepare():
    path = OUT / "protocol.json"
    if path.exists():
        result = json.loads(path.read_text())
        for source, digest in result["input_hashes"].items():
            if file_sha256(Path(source)) != digest: raise ValueError(f"Frozen training input changed: {source}")
        return result
    admission = json.loads(ADMISSION.read_text()); prior = json.loads(PRIOR_PROTOCOL.read_text())
    if admission["status"] != "eligible_for_frozen_development_training" or admission["frame_count"] != 72:
        raise ValueError("Bridge development admission is incomplete")
    if prior["status"] != "frozen_before_training": raise ValueError("Common A-arm source protocol changed")
    prior_ids = set(prior["membership"]["A"]["member_ids"])
    common = [dict(row) for row in prior["pool_rows"] if row["member_id"] in prior_ids]
    if len(common) != 114 or Counter(row["subset"] for row in common) != {"base": 66, "regular": 48}:
        raise ValueError("Common base + regular membership changed")
    for row in common:
        if file_sha256(Path(row["image_path"])) != row["image_sha256"] or file_sha256(Path(row["label_path"])) != row["label_sha256"]:
            raise ValueError("Common member bytes changed")
    OUT.mkdir(parents=True, exist_ok=False); images = OUT / "pool/images"; labels = OUT / "pool/labels"
    images.mkdir(parents=True); labels.mkdir(parents=True)
    added = []
    for index, row in enumerate(admission["entries"]):
        source = Path(row["image_path"]); image_path = images / f"{index:03}.png"; label_path = labels / f"{index:03}.txt"
        with Image.open(source) as image:
            if image.size != (1920, 1080): raise ValueError("Unexpected bridge image size")
            image.convert("RGB").save(image_path)
        if row["subset"] == "bridge_positive":
            ledger = next(item for item in json.loads((BASE / "visual-bridge-supplement-v2/frozen-positive-ledger.json").read_text())["frames"] if item["view_id"] == row["view_id"])
            receipt = json.loads(Path(ledger["receipt_path"]).read_text())
            captured = next(item for item in receipt["views"] if item["view_id"] == row["view_id"])
            truth = captured["truth"]
            if object_sha256(truth) != row["label_sha256"]: raise ValueError("Bridge truth hash changed")
        else:
            truth = {"objects": []}
        label_path.write_text(yolo_lines(truth))
        added.append({"member_id": row["member_id"], "data_role": "new_training_candidate", "subset": row["subset"],
                      "pair_id": row["pair_id"], "derivation_group": row["derivation_group"],
                      "image_path": str(image_path), "image_sha256": file_sha256(image_path),
                      "label_path": str(label_path), "label_sha256": file_sha256(label_path)})
    by_subset = {name: [row for row in added if row["subset"] == name] for name in ("bridge_positive", "hard_negative")}
    arms = {"A": common, "B": common + by_subset["bridge_positive"],
            "C": common + by_subset["hard_negative"], "D": common + added}
    expected = {"A": 114, "B": 162, "C": 138, "D": 186}
    if {arm: len(rows) for arm, rows in arms.items()} != expected: raise ValueError("Arm membership counts changed")
    membership, schedules, exposures, yamls = {}, {}, {}, {}
    all_rows = common + added; by_id = {row["member_id"]: row for row in all_rows}
    for arm, rows in arms.items():
        list_path = OUT / f"arm-{arm}.txt"; list_path.write_text("\n".join(row["image_path"] for row in rows) + "\n")
        yaml_path = OUT / f"dataset-{arm}.yaml"; yaml_path.write_text(f"path: {OUT}\ntrain: {list_path}\nval: {list_path}\nnames: {json.dumps(NAMES)}\n")
        membership[arm] = {"path": str(list_path), "sha256": file_sha256(list_path), "member_ids": [row["member_id"] for row in rows]}
        yamls[arm] = {"path": str(yaml_path), "sha256": file_sha256(yaml_path)}; schedules[arm] = {}; exposures[arm] = {}
        for seed in SEEDS:
            value = schedule(membership[arm]["member_ids"], seed); schedules[arm][str(seed)] = value
            draws = [member for epoch in value for member in epoch]
            exposures[arm][str(seed)] = {"draws": len(draws), "unique_frames": len(set(draws)),
                                         "by_subset": dict(sorted(Counter(by_id[item]["subset"] for item in draws).items()))}
    policy = {"aggregation": "Unweighted mean across seeds 7/17/27; no best-seed selection.",
              "required_all": [
                  {"metric": "original.planned_instance_hit_rate.mean", "op": ">=", "value": 0.8333333333333334},
                  {"metric": "original.planned_instance_hit_rate.min_seed", "op": ">=", "value": 0.75},
                  {"metric": "material.planned_instance_hit_rate.mean", "op": ">=", "value": 0.5},
                  {"metric": "material.planned_instance_hit_rate.min_seed", "op": ">=", "value": 0.3333333333333333},
                  {"metric": "background.planned_instance_hit_rate.mean", "op": ">=", "value": 0.6},
                  {"metric": "lighting.planned_instance_hit_rate.mean", "op": ">=", "value": 0.6},
                  {"metric": "no_target.frame_false_positive_rate.mean", "op": "<=", "value": 0.1},
                  {"metric": "no_target.frame_false_positive_rate.max_seed", "op": "<=", "value": 0.2}],
              "selection_order": "Select D only if all rules pass; otherwise select B only if all rules pass; C is a negative-control diagnosis and A is the common reference. Otherwise select none.",
              "source": "Thresholds carried forward unchanged from the pre-frozen E/F development policy."}
    inputs = (ADMISSION, PRIOR_PROTOCOL, PAIRED_EVAL, NEGATIVE_EVAL, WEIGHTS, Path(__file__))
    result = {"schema_version": 1, "status": "frozen_before_training", "experiment": "visual-bridge-supplement-abcd-v1",
              "arms": {"A": "common base + regular", "B": "A + bridge positives", "C": "A + bridge hard negatives", "D": "A + both"},
              "membership": membership, "pool_rows": all_rows, "dataset_yamls": yamls,
              "schedules": schedules, "exposures": exposures, "seeds": list(SEEDS),
              "controls": {"initial_weights": str(WEIGHTS), "initial_weights_sha256": file_sha256(WEIGHTS), "input_size": 640,
                           "optimizer": "AdamW", "lr0": 0.001, "lrf": 1.0, "epochs": EPOCHS,
                           "slots_per_epoch": SLOTS, "batch": BATCH, "optimizer_steps": 100,
                           "deterministic": True, "augmentation": "disabled", "validation_during_training": False},
              "acceptance_policy": policy, "input_hashes": {str(source): file_sha256(source) for source in inputs},
              "boundaries": ["Development-only comparison; formal training_admitted remains false.",
                             "The fixed 48-frame paired regression never enters training.",
                             "The prior isolated 48-frame no-target regression never enters training.",
                             "The unseen-scene test remains sealed until one candidate family passes every frozen rule."],
              "development_training_eligible": True, "unseen_scene_status": "sealed_not_evaluated",
              "training_admitted": False, "promotable": False}
    result["identity"] = object_sha256(result); write_json(path, result); return result


if __name__ == "__main__":
    value = prepare()
    print(json.dumps({"status": value["status"], "identity": value["identity"],
                      "arm_counts": {arm: len(spec["member_ids"]) for arm, spec in value["membership"].items()},
                      "controls": value["controls"], "acceptance_policy": value["acceptance_policy"]}, indent=2))

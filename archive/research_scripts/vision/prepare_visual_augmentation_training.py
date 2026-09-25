"""Freeze A/B/C/D memberships, training bytes and deterministic schedules."""
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
OUT = BASE / "visual-augmentation-training-v2"
TRUSTED = BASE / "trusted-training-base-v1/frozen-ledger.json"
CANDIDATES = BASE / "visual-augmentation-240-v2/frozen-intake-ledger.json"
POSITIVE_MANIFEST = BASE / "visual-augmentation-240-v1/positive-review-v1/manifest.json"
WEIGHTS = ROOT / "models/equipment/visual-yolo11n-baseline-v2.11-candidate-package-v1/weights/best.pt"
NAMES = ["transformer", "switchgear", "capacitor_bank", "reactor"]
SEEDS = [7, 17, 27]
EPOCHS, SLOTS, BATCH = 10, 60, 6


def read(path):
    return json.loads(path.read_text())


def yolo_lines(truth):
    lines = []
    for obj in truth["objects"]:
        x1,y1,x2,y2 = obj["bbox_xyxy"]
        if obj["class_name"] not in NAMES or not (0 <= x1 < x2 <= 1920 and 0 <= y1 < y2 <= 1080):
            raise ValueError("Invalid full_2d truth object")
        lines.append(f"{NAMES.index(obj['class_name'])} {(x1+x2)/3840:.10f} {(y1+y2)/2160:.10f} {(x2-x1)/1920:.10f} {(y2-y1)/1080:.10f}")
    return "\n".join(lines) + ("\n" if lines else "")


def make_schedule(member_ids, seed):
    rng = random.Random(seed)
    draws = []
    while len(draws) < EPOCHS*SLOTS:
        cycle = list(member_ids); rng.shuffle(cycle); draws.extend(cycle)
    draws = draws[:EPOCHS*SLOTS]
    return [draws[index*SLOTS:(index+1)*SLOTS] for index in range(EPOCHS)]


def exposure(rows, schedule):
    by_id = {row["member_id"]:row for row in rows}
    draws = [member for epoch in schedule for member in epoch]
    return {
        "draws":len(draws), "unique_frames":len(set(draws)),
        "by_data_role":dict(sorted(Counter(by_id[x]["data_role"] for x in draws).items())),
        "by_subset":dict(sorted(Counter(by_id[x]["subset"] for x in draws).items())),
    }


def prepare():
    protocol_path = OUT / "protocol.json"
    if protocol_path.exists():
        protocol = read(protocol_path)
        for path,digest in protocol["input_hashes"].items():
            if file_sha256(Path(path)) != digest:
                raise ValueError(f"Frozen training input changed: {path}")
        if "dataset_yamls" not in protocol:
            dataset_yamls = {}
            for arm in protocol["membership"]:
                path = OUT / f"dataset-{arm}.yaml"
                path.write_text(f"path: {OUT}\ntrain: arm-{arm}.txt\nval: arm-{arm}.txt\nnames: {json.dumps(NAMES)}\n")
                dataset_yamls[arm] = {"path":str(path), "sha256":file_sha256(path)}
            protocol["dataset_yamls"] = dataset_yamls
            protocol.pop("identity", None)
            protocol["identity"] = object_sha256(protocol)
            write_json(protocol_path, protocol)
        return protocol
    trusted, candidates, manifest = read(TRUSTED), read(CANDIDATES), read(POSITIVE_MANIFEST)
    if trusted["status"] != "frozen" or trusted["frame_count"] != 66:
        raise ValueError("Trusted common base is not frozen")
    if candidates["status"] != "frozen" or candidates["counts"]["total"] != 240:
        raise ValueError("Candidate augmentation dataset is not frozen")
    truths = {(row["run_id"],row["view_id"]):row["truth"] for row in manifest["frames"]}
    OUT.mkdir(parents=True, exist_ok=False)
    pool_images = OUT / "pool/images"; pool_labels = OUT / "pool/labels"
    pool_images.mkdir(parents=True); pool_labels.mkdir(parents=True)
    rows = []
    for index, row in enumerate(trusted["entries"]):
        rows.append({"member_id":f"base:{row['view_id']}", "data_role":"trusted_training_base", "subset":"base",
            "image_path":row["training_image_path"], "image_sha256":row["training_image_sha256"],
            "label_path":row["training_label_path"], "label_sha256":row["training_label_sha256"]})
    for index, row in enumerate(candidates["entries"]):
        member_id = f"candidate:{row['candidate_id']}"
        image_path = pool_images / f"{index:03}.png"; label_path = pool_labels / f"{index:03}.txt"
        with Image.open(row["image_path"]) as image:
            if image.size != (1920,1080): raise ValueError("Unexpected candidate image size")
            image.convert("RGB").save(image_path)
        if row["equipment_instance_id"] is None:
            truth = {"objects":[]}
        else:
            truth = truths[(row["recording_group"],row["pose_id"])]
            if object_sha256(truth) != row["label_sha256"]: raise ValueError("Candidate truth hash changed")
        label_path.write_text(yolo_lines(truth))
        subset = "regular" if row["recording_group"] == "regular-positive" else "negative" if row["equipment_instance_id"] is None else "appearance"
        rows.append({"member_id":member_id, "data_role":"new_training_candidate", "subset":subset,
            "source_candidate_id":row["candidate_id"], "derivation_group":row["derivation_group"],
            "image_path":str(image_path), "image_sha256":file_sha256(image_path),
            "label_path":str(label_path), "label_sha256":file_sha256(label_path)})
    by_subset = {name:[row for row in rows if row["subset"] == name] for name in ("base","regular","appearance","negative")}
    arms = {
        "A":by_subset["base"]+by_subset["regular"],
        "B":by_subset["base"]+by_subset["regular"]+by_subset["appearance"],
        "C":by_subset["base"]+by_subset["regular"]+by_subset["negative"],
        "D":by_subset["base"]+by_subset["regular"]+by_subset["appearance"]+by_subset["negative"],
    }
    expected = {"A":114,"B":258,"C":162,"D":306}
    if {k:len(v) for k,v in arms.items()} != expected: raise ValueError("Arm membership counts differ")
    membership, schedules, exposures = {}, {}, {}
    for arm,members in arms.items():
        path = OUT / f"arm-{arm}.txt"; path.write_text("\n".join(row["image_path"] for row in members)+"\n")
        membership[arm] = {"path":str(path), "sha256":file_sha256(path), "member_ids":[row["member_id"] for row in members]}
        schedules[arm] = {}; exposures[arm] = {}
        for seed in SEEDS:
            schedule = make_schedule(membership[arm]["member_ids"], seed)
            schedules[arm][str(seed)] = schedule
            exposures[arm][str(seed)] = exposure(members, schedule)
    dataset = OUT / "dataset.yaml"
    dataset.write_text(f"path: {OUT}\ntrain: arm-A.txt\nval: arm-A.txt\nnames: {json.dumps(NAMES)}\n")
    input_paths = [TRUSTED,CANDIDATES,POSITIVE_MANIFEST,WEIGHTS]
    input_hashes = {str(path):file_sha256(path) for path in input_paths}
    protocol = {
        "schema_version":1, "status":"frozen_before_training", "experiment":"visual-augmentation-abcd-v2",
        "supersedes_unexecuted_protocol":"visual-augmentation-abcd-v1",
        "arms":{"A":"base + regular", "B":"A + appearance", "C":"A + hard negatives", "D":"A + appearance + hard negatives"},
        "membership":membership, "pool_rows":rows, "schedules":schedules, "exposures":exposures,
        "seeds":SEEDS, "controls":{"initial_weights":str(WEIGHTS), "initial_weights_sha256":file_sha256(WEIGHTS),
            "input_size":640, "optimizer":"AdamW", "lr0":0.001, "lrf":1.0, "epochs":EPOCHS,
            "slots_per_epoch":SLOTS, "batch":BATCH, "optimizer_steps":EPOCHS*SLOTS//BATCH,
            "deterministic":True, "augmentation":"disabled", "validation_during_training":False},
        "dataset_yaml":str(dataset), "input_hashes":input_hashes,
        "boundaries":["Development-only comparison; not formal training admission or promotion.",
            "The 48 paired regression frames never enter training.",
            "The unseen-scene test remains sealed until development candidate selection.",
            "Legacy base review validates visible full-frame labels; stable planned-instance identity was not preserved in its old receipts."],
        "training_admitted":False, "promotable":False,
    }
    protocol["identity"] = object_sha256(protocol)
    write_json(protocol_path, protocol)
    return protocol


if __name__ == "__main__":
    result=prepare()
    print(json.dumps({"status":result["status"],"identity":result["identity"],"arm_counts":{k:len(v["member_ids"]) for k,v in result["membership"].items()},"controls":result["controls"],"exposures":result["exposures"]},indent=2))

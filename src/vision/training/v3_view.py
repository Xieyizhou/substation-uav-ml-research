"""Deterministic real/synthetic YOLO11n v3 training-view materialization."""

from __future__ import annotations

from collections import Counter
import hashlib
import json
import os
from pathlib import Path
import shutil
from src.ml import EQUIPMENT_CLASSES
from src.ml.artifacts import file_sha256, object_sha256, write_json
from src.vision.contracts.identity import class_order_identity
from src.vision.contracts.real_domain import RealDomainDatasetIdentity
from src.vision.contracts.training_identity import TrainingViewIdentity
ALGORITHM = "real_synthetic_balanced_v3"
SEED = 7
MAX_TRAIN_FRAMES = 28_000
MAX_REAL_REPETITIONS = 4
PER_CLASS_CAP = 2_800
def _jsonl(path):
    with Path(path).open(encoding="utf-8") as source:
        return [json.loads(line) for line in source if line.strip()]
def _stable(rows, namespace):
    return sorted(rows, key=lambda row: hashlib.sha256(
        f"{SEED}:{namespace}:{row['sample_id']}".encode()
    ).hexdigest())
def _positive_quota(real_rows, synthetic_rows):
    real_positive = [row for row in real_rows if row["classes"]]
    real_union_capacity = len(real_positive) * MAX_REAL_REPETITIONS // 4
    limits = [PER_CLASS_CAP, real_union_capacity]
    for class_name in EQUIPMENT_CLASSES:
        limits.append(
            len([row for row in real_rows if class_name in row["classes"]])
            * MAX_REAL_REPETITIONS
        )
        limits.append(
            len([row for row in synthetic_rows if class_name in row["classes"]])
        )
    return max(0, min(limits))
def _select_balanced(rows, quota, *, source, repeat_limit):
    usage, selected = Counter(), []
    for class_name in EQUIPMENT_CLASSES:
        candidates = _stable(
            [row for row in rows if class_name in row["classes"]],
            f"{source}:{class_name}",
        )
        cursor, accepted = 0, 0
        while accepted < quota and candidates:
            row = candidates[cursor % len(candidates)]
            cursor += 1
            if usage[row["sample_id"]] >= repeat_limit:
                if all(usage[item["sample_id"]] >= repeat_limit for item in candidates):
                    break
                continue
            repeat_index = usage[row["sample_id"]]
            usage[row["sample_id"]] += 1
            selected.append({
                **row,
                "instance_id": f"{source}-{row['sample_id']}-r{repeat_index + 1}",
                "assigned_class": class_name,
                "source_type": source,
                "repeat_index": repeat_index,
            })
            accepted += 1
    counts = Counter(row["assigned_class"] for row in selected)
    if any(counts[name] != quota for name in EQUIPMENT_CLASSES):
        raise ValueError("real-domain samples cannot satisfy balanced v3 class quota")
    return selected, dict(usage)
def _select_negatives(rows, count, *, source, repeat_limit):
    candidates = _stable(
        [row for row in rows if row["annotation_status"] == "verified_no_target"],
        f"{source}:no-target",
    )
    if len(candidates) * repeat_limit < count:
        raise ValueError(f"{source} no-target samples cannot satisfy v3 quota")
    return [
        {
            **candidates[index % len(candidates)],
            "instance_id": f"{source}-{candidates[index % len(candidates)]['sample_id']}"
                           f"-n{index // len(candidates) + 1}",
            "assigned_class": None,
            "source_type": source,
            "repeat_index": index // len(candidates),
        }
        for index in range(count)
    ]
def _real_rows(root):
    identity = RealDomainDatasetIdentity.from_record(json.loads(
        (root / "identity/real_domain_dataset_identity.json").read_text()
    ))
    if not identity.training_eligible:
        raise ValueError("real-domain coverage gate has not passed")
    if file_sha256(root / "identity/samples.jsonl") != identity.managed_manifest_sha256:
        raise ValueError("real-domain managed manifest changed")
    rows = []
    for row in _jsonl(root / "identity/samples.jsonl"):
        rows.append({
            **row,
            "image_path": str(root / row["image_relative_path"]),
            "label_path": str(root / row["label_relative_path"]),
            "annotation_status": (
                "verified_no_target" if row["no_target"] else "labelled"
            ),
        })
    return identity, rows
def _synthetic_rows(root, partition):
    membership = _jsonl(root / "identity" / f"{partition}_membership.jsonl")
    return [{
        **row,
        "image_path": str(root / row["image_relative_path"]),
        "label_path": str(root / row["label_relative_path"]),
        "image_sha256": row["payload_sha256"],
    } for row in membership]
def _link(source, destination):
    destination.parent.mkdir(parents=True, exist_ok=True)
    try:
        os.link(source, destination)
        return "hardlink"
    except OSError:
        shutil.copy2(source, destination)
        return "copy"


def _write_partition(root, name, rows):
    for category in ("images", "labels"):
        path = root / category / name
        if path.exists():
            shutil.rmtree(path)
    membership, modes = [], Counter()
    class_counts, no_target = Counter(), 0
    for row in rows:
        instance = row.get("instance_id", row["sample_id"])
        basename = hashlib.sha256(instance.encode()).hexdigest()
        suffix = Path(row["image_path"]).suffix.lower()
        image_rel = Path("images") / name / f"{basename}{suffix}"
        label_rel = Path("labels") / name / f"{basename}.txt"
        if file_sha256(Path(row["image_path"])) != row["image_sha256"]:
            raise ValueError(f"image hash changed for {row['sample_id']}")
        modes[_link(Path(row["image_path"]), root / image_rel)] += 1
        _link(Path(row["label_path"]), root / label_rel)
        class_counts.update(row["classes"])
        no_target += row["annotation_status"] == "verified_no_target"
        membership.append({
            "sample_id": row["sample_id"], "instance_id": instance,
            "source_type": row.get("source_type", "real_validation"),
            "assigned_class": row.get("assigned_class"),
            "repeat_index": row.get("repeat_index", 0),
            "image_relative_path": image_rel.as_posix(),
            "label_relative_path": label_rel.as_posix(),
            "payload_sha256": row["image_sha256"],
            "label_sha256": file_sha256(root / label_rel),
            "classes": row["classes"],
            "annotation_status": row["annotation_status"],
        })
    path = root / "identity" / f"{name}_membership.jsonl"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(row, sort_keys=True, separators=(",", ":")) + "\n"
                            for row in membership), encoding="utf-8")
    return {
        "membership_sha256": file_sha256(path), "frame_count": len(rows),
        "class_counts": {name: class_counts[name] for name in EQUIPMENT_CLASSES},
        "no_target_count": no_target, "link_modes": dict(modes),
    }


def _yaml(root):
    names = "".join(f"  {index}: {name}\n" for index, name in enumerate(EQUIPMENT_CLASSES))
    (root / "dataset.yaml").write_text(
        f"path: {root.resolve()}\ntrain: images/train\nval: images/validation\n"
        "test: images/full_validation\nnames:\n" + names, encoding="utf-8"
    )
    (root / "dataset-smoke.yaml").write_text(
        f"path: {root.resolve()}\ntrain: images/smoke_train\n"
        "val: images/smoke_validation\nnames:\n" + names, encoding="utf-8"
    )
    (root / "dataset-synthetic-regression.yaml").write_text(
        f"path: {root.resolve()}\nval: images/synthetic_regression\nnames:\n" + names,
        encoding="utf-8",
    )


def materialize_real_training_view(real_dataset_root, synthetic_view_root,
                                   output_root, *, seed=SEED):
    if seed != SEED:
        raise ValueError("real/synthetic v3 sampling seed must be 7")
    real_dataset_root = Path(real_dataset_root)
    synthetic_view_root = Path(synthetic_view_root)
    output_root = Path(output_root)
    if output_root.exists() and any(output_root.iterdir()):
        raise ValueError("v3 training-view output must be empty")
    output_root.mkdir(parents=True, exist_ok=True)
    real_identity, real = _real_rows(real_dataset_root)
    real_development = [row for row in real if row["partition"] == "development"]
    real_validation = [row for row in real if row["partition"] == "validation"]
    synthetic_train = _synthetic_rows(synthetic_view_root, "train")
    synthetic_validation = _synthetic_rows(synthetic_view_root, "full_validation")
    quota = _positive_quota(real_development, synthetic_train)
    if quota <= 0:
        raise ValueError("v3 positive class quota is empty")
    real_positive, repeat_counts = _select_balanced(
        real_development, quota, source="real", repeat_limit=MAX_REAL_REPETITIONS
    )
    synthetic_positive, _ = _select_balanced(
        synthetic_train, quota, source="synthetic", repeat_limit=1
    )
    real_negative = _select_negatives(
        real_development, quota, source="real", repeat_limit=MAX_REAL_REPETITIONS
    )
    synthetic_negative = _select_negatives(
        synthetic_train, quota, source="synthetic", repeat_limit=1
    )
    train = _stable(
        real_positive + synthetic_positive + real_negative + synthetic_negative,
        "train-order",
    )
    if len(train) > MAX_TRAIN_FRAMES:
        raise AssertionError("v3 training view exceeds its fixed frame budget")
    validation = _stable(real_validation, "real-validation")
    smoke_train = train[:min(256, len(train))]
    smoke_validation = validation[:min(64, len(validation))]
    results = {
        "train": _write_partition(output_root, "train", train),
        "validation": _write_partition(output_root, "validation", validation),
        "full_validation": _write_partition(output_root, "full_validation", validation),
        "synthetic_regression": _write_partition(
            output_root, "synthetic_regression", synthetic_validation
        ),
        "smoke_train": _write_partition(output_root, "smoke_train", smoke_train),
        "smoke_validation": _write_partition(
            output_root, "smoke_validation", smoke_validation
        ),
    }
    labels_manifest = write_json(output_root / "identity/labels_manifest.json", {
        name: {"membership_sha256": value["membership_sha256"],
               "frame_count": value["frame_count"]}
        for name, value in results.items()
    })
    source_synthetic = json.loads(
        (synthetic_view_root / "identity/training_view_identity.json").read_text()
    )
    identity = TrainingViewIdentity(
        source_development_dataset_identity=(
            real_identity.real_domain_dataset_identity_sha256
        ),
        source_validation_dataset_identity=(
            real_identity.real_domain_dataset_identity_sha256
        ),
        sampling_algorithm=ALGORITHM,
        sampling_seed=SEED,
        train_membership_sha256=results["train"]["membership_sha256"],
        validation_membership_sha256=results["validation"]["membership_sha256"],
        full_validation_membership_sha256=results["full_validation"]["membership_sha256"],
        labels_manifest_sha256=file_sha256(labels_manifest),
        class_order_identity=class_order_identity(),
        train_frame_count=results["train"]["frame_count"],
        validation_frame_count=results["validation"]["frame_count"],
        full_validation_frame_count=results["full_validation"]["frame_count"],
        train_class_counts=results["train"]["class_counts"],
        validation_class_counts=results["validation"]["class_counts"],
        train_no_target_count=results["train"]["no_target_count"],
        validation_no_target_count=results["validation"]["no_target_count"],
    )
    identity_path = write_json(
        output_root / "identity/training_view_identity.json", identity.to_record()
    )
    v3_record = {
        "real_training_view_schema_version": 1,
        "sampling_algorithm": ALGORITHM,
        "sampling_seed": SEED,
        "real_domain_dataset_identity_sha256": (
            real_identity.real_domain_dataset_identity_sha256
        ),
        "synthetic_training_view_identity_sha256": source_synthetic[
            "training_view_identity_sha256"
        ],
        "positive_quota_per_class_per_source": quota,
        "maximum_real_repetitions": MAX_REAL_REPETITIONS,
        "real_repeat_counts": dict(sorted(repeat_counts.items())),
        "partitions": results,
        "blind_membership_included": False,
    }
    v3_record["real_training_view_identity_sha256"] = object_sha256(v3_record)
    write_json(output_root / "identity/real_training_view_identity.json", v3_record)
    _yaml(output_root)
    return {"identity": identity.to_record(), "identity_path": str(identity_path),
            "real_training_identity": v3_record,
            "dataset_yaml": str(output_root / "dataset.yaml"),
            "smoke_dataset_yaml": str(output_root / "dataset-smoke.yaml")}

"""Read identity-bound dataset membership without loading training libraries."""

import json
from pathlib import Path

from src.ml import EQUIPMENT_CLASSES
from src.ml.artifacts import file_sha256, object_sha256
from src.sandbox.feedback_review import checked_json, _inside


def dataset_members(dataset):
    root = Path(dataset.dataset_root)
    if dataset.source_type == "native_training_view":
        identity = checked_json(root / "identity/training_view_identity.json", "training_view_identity_sha256")
        if identity["training_view_identity_sha256"] != dataset.dataset_identity_sha256:
            raise ValueError("Base dataset identity changed")
        members = []
        for split in ("train", "validation"):
            path = root / "identity" / (split + "_membership.jsonl")
            if file_sha256(path) != identity[split + "_membership_sha256"]:
                raise ValueError("Base dataset membership changed")
            for line in path.read_text().splitlines():
                if not line.strip():
                    continue
                row = json.loads(line)
                members.append(dict(sample_id=row["sample_id"], split=split,
                    image_path=row["image_relative_path"], label_path=row["label_relative_path"],
                    image_sha256=row.get("image_sha256", row.get("payload_sha256")), label_sha256=row["label_sha256"],
                    source_group=row.get("source_collection_identity", row.get("recording_id", dataset.dataset_identity_sha256 + ":" + split))))
        return members
    members = json.loads((root / "membership.json").read_text())
    payload = dict(members=members, class_names=EQUIPMENT_CLASSES)
    if dataset.source_type == "reviewed_feedback":
        provenance = json.loads((root / "provenance.json").read_text())
        if object_sha256(provenance) != dataset.source_identity_sha256:
            raise ValueError("Reviewed dataset provenance changed")
        payload["provenance_identity_sha256"] = dataset.source_identity_sha256
    elif dataset.source_type != "imported_yolo":
        raise ValueError("Unsupported base dataset source")
    if object_sha256(payload) != dataset.dataset_identity_sha256:
        raise ValueError("Dataset membership identity changed")
    rows = []
    for member in members:
        row = dict(member)
        if row["split"] not in ("train", "validation"):
            raise ValueError("Dataset contains an unsupported split")
        if "image_path" not in row:
            name = row["sample_id"]
            if Path(name).name != name or name in ("", ".", ".."):
                raise ValueError("Invalid base sample identifier")
            images = list((root / "images" / row["split"]).glob(name + ".*"))
            if len(images) != 1:
                raise ValueError("Base sample image is missing or ambiguous")
            row.update(image_path=images[0].relative_to(root).as_posix(),
                       label_path=f"labels/{row['split']}/{name}.txt")
        row.setdefault("source_group", dataset.dataset_identity_sha256 + ":" + row["split"])
        rows.append(row)
    return rows


def member_paths(root, member):
    root = Path(root)
    paths = tuple(_inside(root, root / member[field + "_path"]) for field in ("image", "label"))
    for field, path in zip(("image", "label"), paths):
        if file_sha256(path) != member[field + "_sha256"]:
            raise ValueError("Dataset " + field + " changed: " + str(path))
    return paths


def verify_reviewed_dataset(dataset):
    root = Path(dataset.dataset_root)
    members = dataset_members(dataset)
    expected = set()
    groups = {"train": set(), "validation": set()}
    hashes = set()
    for member in members:
        paths = member_paths(root, member)
        expected.update(str(path.resolve()) for path in paths)
        groups[member["split"]].add(member["source_group"])
        if member["image_sha256"] in hashes:
            raise ValueError("Reviewed dataset contains duplicate images")
        hashes.add(member["image_sha256"])
    actual = {str(p.resolve()) for folder in ("images", "labels") for p in (root / folder).rglob("*")
              if p.is_file() and p.suffix.lower() in (".png", ".jpg", ".jpeg", ".txt")}
    if actual != expected or groups["train"] & groups["validation"]:
        raise ValueError("Reviewed dataset membership or source isolation changed")
    return members

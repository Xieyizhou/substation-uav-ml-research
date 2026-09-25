"""Register reviewed feedback as training increments with fixed base validation."""

from collections import Counter
from contextlib import ExitStack
import json
from pathlib import Path
import shutil

from src.ml import EQUIPMENT_CLASSES
from src.ml.artifacts import file_sha256, object_sha256, write_json
from src.sandbox.feedback_review import FeedbackReviewStore, _inside, validate_decision
from src.sandbox.workbench_datasets import resolve_workbench_dataset, _validate_label, _write_dataset_yaml
from src.sandbox.workbench_membership import dataset_members, member_paths, verify_reviewed_dataset
from src.sandbox.workbench_models import WorkbenchDataset
from src.sandbox.workbench_recipe import IDENTIFIER

MAX_FEEDBACK = 512
MAX_BYTES = 512 * 1024**2


def _evenly(rows, limit):
    rows = sorted(rows, key=lambda row: row["image_path"])
    return rows if len(rows) <= limit else [rows[((2*i+1)*len(rows))//(2*limit)] for i in range(limit)]


def _review_label(review, frame):
    fields = {key: review[key] for key in ("decision", "boxes", "no_target", "complete_annotation", "reviewer", "note")}
    validate_decision(fields, frame)
    lines = []
    for box in review["boxes"]:
        x1, y1, x2, y2 = box["xyxy"]
        values = ((x1+x2)/2/frame["width"], (y1+y2)/2/frame["height"],
                  (x2-x1)/frame["width"], (y2-y1)/frame["height"])
        lines.append(str(EQUIPMENT_CLASSES.index(box["class_name"])) + " " + " ".join(f"{v:.10f}" for v in values))
    return "\n".join(lines) + ("\n" if lines else "")


def register_feedback_dataset(config, dataset_id, base_dataset_id, collection_ids, *, base_train_limit=256, validation_limit=64):
    if config.profile != "development":
        raise ValueError("Feedback registration requires Development profile")
    if not IDENTIFIER.fullmatch(str(dataset_id)):
        raise ValueError("Dataset ID must use lowercase letters, numbers and hyphens")
    if not isinstance(collection_ids, list) or not 1 <= len(collection_ids) <= 16 or len(set(collection_ids)) != len(collection_ids):
        raise ValueError("Select 1–16 distinct completed feedback collections")
    if base_train_limit not in (128, 256, 512) or validation_limit not in (64, 128, 256):
        raise ValueError("Unsupported base replay or validation budget")
    base = resolve_workbench_dataset(config.project_root, config.workbench_datasets_root, base_dataset_id)
    base_root = _inside(config.project_root, Path(base.dataset_root))
    rows = dataset_members(base)
    selected = [row for split, limit in (("train", base_train_limit), ("validation", validation_limit))
                for row in _evenly([r for r in rows if r["split"] == split], limit)]
    if {row["split"] for row in selected} != {"train", "validation"}:
        raise ValueError("Base dataset requires both training and validation")
    store = FeedbackReviewStore(config)
    snapshots, feedback = [], []
    with ExitStack() as stack:
        for collection_id in collection_ids:
            directory, summary, members = stack.enter_context(store.completed(collection_id))
            reviews = []
            for sample_id, _ in members:
                sample, image = store.sample(collection_id, sample_id, directory=directory)
                review = store.review(collection_id, sample_id, sample)
                if review is None:
                    raise ValueError("Collection has unreviewed samples; accept or reject every sample first")
                reviews.append(review)
                if review["decision"] == "accepted":
                    feedback.append((sample, image, review))
            snapshots.append(dict(collection_id=collection_id, summary=summary, reviews=reviews))
        if not 1 <= len(feedback) <= MAX_FEEDBACK:
            raise ValueError("Registration requires 1–512 explicitly accepted feedback images")
        provenance = dict(schema_version=1, base_dataset_id=base.dataset_id,
            base_dataset_identity_sha256=base.dataset_identity_sha256,
            split_policy="feedback_sources_train_only_fixed_base_validation",
            base_train_limit=base_train_limit, validation_limit=validation_limit,
            collection_snapshots=snapshots, training_admitted=True,
            scope="development increment; no automatic model promotion")
        destination = _inside(config.workbench_datasets_root, config.workbench_datasets_root / dataset_id)
        destination.mkdir(parents=True, exist_ok=False)
        try:
            result = _materialize(destination, base_root, selected, feedback, provenance, dataset_id)
            verify_reviewed_dataset(WorkbenchDataset.from_record(result))
            return result
        except BaseException:
            shutil.rmtree(destination)
            raise


def _materialize(destination, base_root, selected, feedback, provenance, dataset_id):
    from PIL import Image

    members, hashes, pixels = [], set(), set()
    counts = {split: Counter() for split in ("train", "validation")}
    negatives = Counter()
    total_bytes = 0

    def append(image, label, split, origin, expected_hash):
        nonlocal total_bytes
        with Image.open(image) as opened:
            rgb = opened.convert("RGB")
            from src.sensors.camera_decoded import decoded_content_sha256
            import numpy as np
            pixel_hash = decoded_content_sha256(np.asarray(rgb))
        digest = file_sha256(image)
        if digest != expected_hash:
            raise ValueError("Source image changed during registration")
        if digest in hashes or pixel_hash in pixels:
            raise ValueError("Duplicate image or decoded pixels across dataset membership")
        hashes.add(digest); pixels.add(pixel_hash)
        total_bytes += image.stat().st_size + len(label.encode())
        if total_bytes > MAX_BYTES:
            raise ValueError("Reviewed dataset exceeds 512 MiB budget")
        sample_id = f"{split}-{len(members):06d}"
        image_path = f"images/{split}/{sample_id}{image.suffix.lower()}"
        label_path = f"labels/{split}/{sample_id}.txt"
        for path in (image_path, label_path):
            (destination / path).parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(image, destination / image_path)
        if file_sha256(destination / image_path) != digest:
            raise ValueError("Image changed while copying")
        (destination / label_path).write_text(label)
        mapping = dict(enumerate(EQUIPMENT_CLASSES))
        objects, _ = _validate_label(destination / label_path, mapping, mapping)
        counts[split].update(objects); negatives[split] += int(not objects)
        members.append(dict(sample_id=sample_id, split=split, image_path=image_path, label_path=label_path,
            image_sha256=digest, rgb_sha256=pixel_hash, label_sha256=file_sha256(destination / label_path), **origin))

    for row in selected:
        image, label = member_paths(base_root, row)
        append(image, label.read_text(), row["split"], dict(source_group=row["source_group"],
            annotation_source="base_dataset", base_sample_id=row["sample_id"]), row["image_sha256"])
    for sample, image, review in feedback:
        append(image, _review_label(review, sample["frame"]), "train", dict(source_group=review["source_group"],
            annotation_source="explicit_review", review_identity_sha256=review["review_identity_sha256"]), review["image_sha256"])
    missing = set(EQUIPMENT_CLASSES) - set(counts["validation"])
    if missing:
        raise ValueError("Fixed validation subset lacks classes: " + ", ".join(sorted(missing)))
    source_identity = object_sha256(provenance)
    identity = object_sha256(dict(members=members, class_names=EQUIPMENT_CLASSES, provenance_identity_sha256=source_identity))
    dataset = WorkbenchDataset(dataset_id=dataset_id, source_type="reviewed_feedback", dataset_root=str(destination),
        dataset_identity_sha256=identity, source_identity_sha256=source_identity,
        split_counts=dict(Counter(row["split"] for row in members)),
        class_counts={key: dict(value) for key, value in counts.items()}, no_target_counts=dict(negatives))
    write_json(destination / "membership.json", members)
    write_json(destination / "provenance.json", provenance)
    _write_dataset_yaml(destination)
    write_json(destination / "dataset.json", dataset.to_record())
    return dataset.to_record()

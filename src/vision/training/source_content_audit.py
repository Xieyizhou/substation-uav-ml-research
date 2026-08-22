"""Stream image content from quarantined YOLO ZIPs and audit uniqueness."""

from __future__ import annotations

from collections import defaultdict
from hashlib import sha256
from io import BytesIO
import json
from pathlib import Path, PurePosixPath
import re
import zipfile

from PIL import Image, UnidentifiedImageError

from src.ml import EQUIPMENT_CLASSES
from src.ml.artifacts import file_sha256, object_sha256, write_json
from src.vision.training.source_export_audit import (
    IMAGE_SUFFIXES, _class_names, _load_dataset_yaml, _safe_entries, _split_for,
)
from src.vision.training.source_feasibility import load_source_registry


CONTENT_AUDIT_SCHEMA_VERSION = 1
LINEAGE = re.compile(r"^(?P<stem>.+?)\.rf\.[a-f0-9]+$", re.I)


def audit_source_content(registry_path, source_id, archive_path, output_root):
    """Write a deterministic content receipt without extracting the archive."""
    _, sources = load_source_registry(registry_path)
    if source_id not in sources:
        raise ValueError(f"unknown real-domain source: {source_id}")
    source = sources[source_id]
    if source["ingestion_status"] != "quarantine":
        raise ValueError("content audit is limited to quarantine sources")
    archive = Path(archive_path)
    destination = Path(output_root) / source_id / "content-audit.json"
    if destination.exists():
        raise FileExistsError(f"content audit already exists: {destination}")
    with zipfile.ZipFile(archive) as bundle:
        entries = _safe_entries(bundle)
        names = _class_names(_load_dataset_yaml(bundle, entries)[0])
        labels = _labels_by_key(bundle, entries, names)
        records = _image_records(bundle, entries, labels)
    decisions = _semantic_decisions(source, names)
    receipt = _receipt(source_id, archive, records, decisions)
    write_json(destination, receipt)
    _refresh_cross_source(Path(output_root))
    return json.loads(destination.read_text(encoding="utf-8"))


def _labels_by_key(bundle, entries, names):
    labels = defaultdict(set)
    for entry in entries:
        path = PurePosixPath(entry.filename)
        split = _split_for(path)
        if split is None or path.suffix.lower() != ".txt" or "labels" not in path.parts:
            continue
        for line in bundle.read(entry).decode("utf-8").splitlines():
            fields = line.split()
            if fields:
                class_id = int(float(fields[0]))
                if not 0 <= class_id < len(names):
                    raise ValueError("content audit found an invalid class id")
                labels[(split, path.stem)].add(names[class_id])
    return labels


def _image_records(bundle, entries, labels):
    records = []
    for entry in entries:
        path = PurePosixPath(entry.filename)
        split = _split_for(path)
        if split is None or path.suffix.lower() not in IMAGE_SUFFIXES or "images" not in path.parts:
            continue
        payload = bundle.read(entry)
        try:
            with Image.open(BytesIO(payload)) as image:
                image.load()
                width, height = image.size
                gray = image.convert("L").resize((9, 8))
                pixels = list(gray.getdata())
        except (OSError, UnidentifiedImageError) as error:
            raise ValueError(f"image is not decodable: {entry.filename}") from error
        bits = [pixels[y * 9 + x] > pixels[y * 9 + x + 1] for y in range(8) for x in range(8)]
        perceptual = sum(int(bit) << index for index, bit in enumerate(bits))
        match = LINEAGE.match(path.stem)
        records.append({
            "path": entry.filename, "split": split, "sha256": sha256(payload).hexdigest(),
            "perceptual_hash": f"{perceptual:016x}", "width": width, "height": height,
            "lineage_id": match.group("stem") if match else path.stem,
            "source_labels": sorted(labels.get((split, path.stem), set())),
        })
    return sorted(records, key=lambda row: row["path"])


def _semantic_decisions(source, names):
    reverse = defaultdict(list)
    for target, mapping in source.get("class_candidates", {}).items():
        status = "accepted" if mapping["semantic_status"] == "verified" else "ambiguous"
        for label in mapping.get("source_labels", []):
            reverse[label].append({"target_class": target, "status": status})
    return {
        status: [
            {"source_label": label, **decision}
            for label in sorted(names) for decision in (
                reverse.get(label) or [{"target_class": None, "status": "ignored"}]
            ) if decision["status"] == status
        ]
        for status in ("accepted", "ambiguous", "ignored")
    }


def _receipt(source_id, archive, records, decisions):
    digest_groups = defaultdict(list)
    lineage_groups = defaultdict(list)
    for row in records:
        digest_groups[row["sha256"]].append(row)
        lineage_groups[row["lineage_id"]].append(row["path"])
    exact = [group for group in digest_groups.values() if len(group) > 1]
    leakage = [group for group in exact if len({row["split"] for row in group}) > 1]
    near = []
    for index, left in enumerate(records):
        for right in records[index + 1:]:
            if left["sha256"] == right["sha256"]:
                continue
            distance = (int(left["perceptual_hash"], 16) ^ int(right["perceptual_hash"], 16)).bit_count()
            if distance <= 3:
                near.append({"left": left["path"], "right": right["path"],
                             "distance": distance, "cross_split": left["split"] != right["split"]})
    aliases = {
        item["source_label"]: item["target_class"]
        for status in ("accepted", "ambiguous") for item in decisions[status]
    }
    unique_classes = defaultdict(set)
    for digest, group in digest_groups.items():
        for label in {label for row in group for label in row["source_labels"]}:
            if label in aliases:
                unique_classes[aliases[label]].add(digest)
    suspicious_lineages = [paths for paths in lineage_groups.values() if len(paths) > 1]
    export_path = archive.parent.parent / "export-audit.json"
    export = json.loads(export_path.read_text(encoding="utf-8")) if export_path.is_file() else {}
    declared_augmentation = export.get("augmentation_declared")
    receipt = {
        "source_content_audit_schema_version": CONTENT_AUDIT_SCHEMA_VERSION,
        "source_id": source_id, "archive_name": archive.name,
        "archive_sha256": file_sha256(archive), "image_count": len(records),
        "independent_image_count": len(digest_groups),
        "candidate_unique_image_count": {
            name: len(unique_classes[name]) for name in EQUIPMENT_CLASSES
        },
        "exact_duplicate_cluster_count": len(exact),
        "exact_duplicate_clusters": [[row["path"] for row in group] for group in exact[:50]],
        "cross_split_leakage_count": len(leakage),
        "cross_split_leakage_clusters": [[row["path"] for row in group] for group in leakage[:50]],
        "near_duplicate_pair_count": len(near),
        "near_duplicate_examples": near[:50],
        "cross_split_near_duplicate_count": sum(item["cross_split"] for item in near),
        "lineage_cluster_count": len(suspicious_lineages),
        "augmentation_declared": declared_augmentation,
        "augmentation_evidence_detected": bool(suspicious_lineages),
        "augmentation_declaration_conflict": (
            declared_augmentation is False and bool(suspicious_lineages)
        ),
        "site_scene_review_queue": suspicious_lineages[:100],
        "semantic_review_manifest": decisions,
        "image_records": records,
        "comparison_scope": [], "cross_source_duplicate_count": 0,
        "cross_source_near_duplicate_count": 0,
        "cross_source_duplicate_examples": [],
        "training_eligible": False,
    }
    receipt["content_audit_identity_sha256"] = object_sha256(receipt)
    return receipt


def _refresh_cross_source(root):
    paths = sorted(root.glob("*/content-audit.json"))
    receipts = [(path, json.loads(path.read_text(encoding="utf-8"))) for path in paths]
    digest_sources = defaultdict(list)
    for _, receipt in receipts:
        for row in receipt["image_records"]:
            digest_sources[row["sha256"]].append((receipt["source_id"], row["path"]))
    shared = {digest: values for digest, values in digest_sources.items()
              if len({source for source, _ in values}) > 1}
    all_sources = sorted(receipt["source_id"] for _, receipt in receipts)
    cross_near = defaultdict(list)
    flattened = [
        (receipt["source_id"], row) for _, receipt in receipts for row in receipt["image_records"]
    ]
    for index, (left_source, left) in enumerate(flattened):
        for right_source, right in flattened[index + 1:]:
            if left_source == right_source or left["sha256"] == right["sha256"]:
                continue
            distance = (int(left["perceptual_hash"], 16) ^ int(right["perceptual_hash"], 16)).bit_count()
            if distance <= 3:
                item = {"left_source": left_source, "left": left["path"],
                        "right_source": right_source, "right": right["path"],
                        "distance": distance}
                cross_near[left_source].append(item)
                cross_near[right_source].append(item)
    for path, receipt in receipts:
        examples = [
            {"sha256": digest, "matches": values}
            for digest, values in sorted(shared.items())
            if any(source == receipt["source_id"] for source, _ in values)
        ]
        receipt["comparison_scope"] = all_sources
        receipt["cross_source_duplicate_count"] = len(examples)
        receipt["cross_source_duplicate_examples"] = examples[:50]
        receipt["cross_source_near_duplicate_count"] = len(cross_near[receipt["source_id"]])
        receipt["cross_source_near_duplicate_examples"] = cross_near[receipt["source_id"]][:50]
        receipt.pop("content_audit_identity_sha256", None)
        receipt["content_audit_identity_sha256"] = object_sha256(receipt)
        write_json(path, receipt)

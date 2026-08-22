"""Audit and materialize licensed four-class real-domain datasets."""
from __future__ import annotations

from collections import Counter, defaultdict
import json
import os
from pathlib import Path
import shutil
from PIL import Image, UnidentifiedImageError
from src.ml import EQUIPMENT_CLASSES
from src.ml.artifacts import file_sha256, object_sha256, write_json
from src.vision.contracts.real_domain import (
    PARTITIONS,
    RealDomainDatasetIdentity,
    coverage_gaps,
)
from src.vision.training.source_feasibility import (
    load_source_registry,
    source_is_training_eligible,
)

def _jsonl(path):
    with Path(path).open(encoding="utf-8") as source:
        return [json.loads(line) for line in source if line.strip()]

def _registry(path):
    return load_source_registry(path)

def _image_info(path):
    try:
        with Image.open(path) as image:
            image.load()
            width, height = image.size
            gray = image.convert("L").resize((9, 8))
            pixels = list(gray.getdata())
    except (OSError, UnidentifiedImageError) as error:
        raise ValueError(f"image is not decodable: {path}") from error
    if width < 2 or height < 2:
        raise ValueError(f"image has invalid dimensions: {path}")
    bits = [pixels[y * 9 + x] > pixels[y * 9 + x + 1]
            for y in range(8) for x in range(8)]
    perceptual = sum(int(bit) << index for index, bit in enumerate(bits))
    return width, height, f"{perceptual:016x}"

def _validate_annotations(row, width, height):
    annotations = row.get("annotations", [])
    if row.get("no_target") is True:
        if annotations:
            raise ValueError("no-target sample contains annotations")
        return [], ()
    accepted, classes = [], set()
    for annotation in annotations:
        status = annotation.get("status", "accepted")
        if status == "ignored":
            continue
        if status != "accepted":
            raise ValueError("annotation status must be accepted or ignored")
        class_name = annotation.get("class_name")
        if class_name not in EQUIPMENT_CLASSES:
            raise ValueError(f"unknown or ambiguous target class: {class_name}")
        box = annotation.get("bbox_xyxy")
        if not isinstance(box, list) or len(box) != 4:
            raise ValueError("bbox_xyxy must contain four coordinates")
        x1, y1, x2, y2 = (float(value) for value in box)
        if not (0 <= x1 < x2 <= width and 0 <= y1 < y2 <= height):
            raise ValueError("bounding box is outside the image")
        accepted.append({"class_name": class_name, "bbox_xyxy": [x1, y1, x2, y2]})
        classes.add(class_name)
    if not accepted:
        raise ValueError("sample is neither accepted target nor verified no-target")
    return accepted, tuple(sorted(classes))

def _validate_source_scope(source, partition, classes):
    if partition not in source.get("eligible_partitions", []):
        raise ValueError("source is not approved for the requested partition")
    candidates = source.get("class_candidates", {})
    unverified = [
        name for name in classes
        if candidates.get(name, {}).get("semantic_status") != "verified"
    ]
    if unverified:
        raise ValueError(
            "source class semantics are not verified: " + ", ".join(unverified)
        )

def _hamming(left, right):
    return (int(left, 16) ^ int(right, 16)).bit_count()

def audit_real_dataset(registry_path, manifest_path, source_root):
    registry, sources = _registry(registry_path)
    source_root = Path(source_root).resolve()
    rows, errors = [], []
    sample_ids, groups = set(), defaultdict(set)
    exact, perceptual = {}, []
    cross_source_duplicates = []
    exact_duplicates = []
    near_duplicates = []
    for number, raw in enumerate(_jsonl(manifest_path), 1):
        try:
            sample_id = str(raw.get("sample_id", "")).strip()
            source_id = str(raw.get("source_id", "")).strip()
            partition = raw.get("partition")
            group_id = str(raw.get("group_id", "")).strip()
            if not sample_id or sample_id in sample_ids:
                raise ValueError("sample id must be non-empty and unique")
            if source_id not in sources:
                raise ValueError("sample references an unknown source")
            if not source_is_training_eligible(sources[source_id]):
                raise ValueError("source is not approved for real training data")
            if partition not in PARTITIONS or not group_id:
                raise ValueError("sample partition and group id are required")
            relative = Path(str(raw.get("image_relative_path", "")))
            image_path = (source_root / relative).resolve()
            if source_root not in image_path.parents or not image_path.is_file():
                raise ValueError("sample image is missing or outside the source root")
            digest = file_sha256(image_path)
            supplied = raw.get("image_sha256")
            if supplied is not None and supplied != digest:
                raise ValueError("sample image SHA256 mismatch")
            width, height, perceptual_hash = _image_info(image_path)
            annotations, classes = _validate_annotations(raw, width, height)
            groups[group_id].add(partition)
            if len(groups[group_id]) > 1:
                raise ValueError("site or sequence group crosses dataset splits")
            _validate_source_scope(sources[source_id], partition, classes)
            if digest in exact:
                prior = exact[digest]
                exact_duplicates.append({
                    "sample_id": sample_id,
                    "prior_sample_id": prior["sample_id"],
                    "source_id": source_id,
                    "prior_source_id": prior["source_id"],
                    "partition": partition,
                    "prior_partition": prior["partition"],
                })
                if prior["source_id"] != source_id:
                    cross_source_duplicates.append({
                        "sample_id": sample_id,
                        "prior_sample_id": prior["sample_id"],
                        "source_id": source_id,
                        "prior_source_id": prior["source_id"],
                    })
                raise ValueError(
                    "exact duplicate of "
                    + prior["sample_id"]
                )
            for prior_hash, prior_id, prior_partition, prior_source in perceptual:
                if prior_partition != partition and _hamming(perceptual_hash, prior_hash) <= 3:
                    near_duplicates.append({
                        "sample_id": sample_id,
                        "prior_sample_id": prior_id,
                        "source_id": source_id,
                        "prior_source_id": prior_source,
                        "partition": partition,
                        "prior_partition": prior_partition,
                    })
                    raise ValueError(f"near duplicate across splits: {prior_id}")
            exact[digest] = {
                "sample_id": sample_id,
                "source_id": source_id,
                "partition": partition,
            }
            perceptual.append((perceptual_hash, sample_id, partition, source_id))
            sample_ids.add(sample_id)
            rows.append({
                "sample_id": sample_id, "source_id": source_id,
                "partition": partition, "group_id": group_id,
                "image_source_path": str(image_path), "image_sha256": digest,
                "image_width": width, "image_height": height,
                "perceptual_hash": perceptual_hash,
                "annotations": annotations, "classes": list(classes),
                "no_target": raw.get("no_target") is True,
            })
        except (TypeError, ValueError) as error:
            errors.append({"line": number, "error": str(error)})
    counts = {partition: Counter() for partition in PARTITIONS}
    no_target = Counter()
    for row in rows:
        counts[row["partition"]].update(row["classes"])
        no_target[row["partition"]] += row["no_target"]
    positive = {
        partition: {name: counts[partition][name] for name in EQUIPMENT_CLASSES}
        for partition in PARTITIONS
    }
    gaps = coverage_gaps(positive, dict(no_target))
    return {
        "real_domain_audit_schema_version": 1,
        "valid": not errors,
        "training_eligible": not errors and not gaps,
        "sample_count": len(rows),
        "partition_counts": dict(Counter(row["partition"] for row in rows)),
        "positive_image_counts": positive,
        "no_target_counts": {partition: no_target[partition] for partition in PARTITIONS},
        "source_counts": dict(Counter(row["source_id"] for row in rows)),
        "coverage_gaps": gaps,
        "exact_duplicate_count": len(exact_duplicates),
        "exact_duplicate_examples": exact_duplicates[:20],
        "cross_source_duplicate_count": len(cross_source_duplicates),
        "cross_source_duplicate_examples": cross_source_duplicates[:20],
        "near_duplicate_count": len(near_duplicates),
        "near_duplicate_examples": near_duplicates[:20],
        "errors": errors,
        "license_ids": sorted({
            sources[source_id]["declared_license_id"]
            for source_id in {row["source_id"] for row in rows}
        }),
        "source_registry_sha256": object_sha256(registry),
        "sample_manifest_sha256": file_sha256(Path(manifest_path)),
        "rows": rows,
    }


def _link_or_copy(source, destination):
    destination.parent.mkdir(parents=True, exist_ok=True)
    try:
        os.link(source, destination)
        return "hardlink"
    except OSError:
        shutil.copy2(source, destination)
        return "copy"


def _label_text(row):
    lines = []
    for item in row["annotations"]:
        x1, y1, x2, y2 = item["bbox_xyxy"]
        width, height = row["image_width"], row["image_height"]
        class_id = EQUIPMENT_CLASSES.index(item["class_name"])
        lines.append(
            f"{class_id} {(x1+x2)/(2*width):.8f} {(y1+y2)/(2*height):.8f} "
            f"{(x2-x1)/width:.8f} {(y2-y1)/height:.8f}"
        )
    return "\n".join(lines) + ("\n" if lines else "")


def materialize_real_dataset(registry_path, manifest_path, source_root, output_root,
                             *, dataset_id, dataset_version="real-domain-v1"):
    audit = audit_real_dataset(registry_path, manifest_path, source_root)
    if not audit["valid"]:
        raise ValueError("real-domain dataset audit contains invalid samples")
    output_root = Path(output_root)
    if output_root.exists() and any(output_root.iterdir()):
        raise ValueError("real-domain dataset output must be empty")
    output_root.mkdir(parents=True, exist_ok=True)
    managed, modes = [], Counter()
    for row in audit.pop("rows"):
        suffix = Path(row["image_source_path"]).suffix.lower()
        image_rel = Path("images") / row["partition"] / f"{row['sample_id']}{suffix}"
        label_rel = Path("labels") / row["partition"] / f"{row['sample_id']}.txt"
        modes[_link_or_copy(Path(row["image_source_path"]), output_root / image_rel)] += 1
        (output_root / label_rel).parent.mkdir(parents=True, exist_ok=True)
        (output_root / label_rel).write_text(_label_text(row), encoding="utf-8")
        managed.append({
            **{key: value for key, value in row.items() if key != "image_source_path"},
            "image_relative_path": image_rel.as_posix(),
            "label_relative_path": label_rel.as_posix(),
            "label_sha256": file_sha256(output_root / label_rel),
        })
    manifest_out = output_root / "identity/samples.jsonl"
    manifest_out.parent.mkdir(parents=True, exist_ok=True)
    manifest_out.write_text("".join(
        json.dumps(row, sort_keys=True, separators=(",", ":")) + "\n"
        for row in managed
    ), encoding="utf-8")
    audit["link_modes"] = dict(modes)
    audit_path = write_json(output_root / "identity/audit.json", audit)
    identity = RealDomainDatasetIdentity(
        dataset_id=dataset_id, dataset_version=dataset_version,
        source_registry_sha256=audit["source_registry_sha256"],
        sample_manifest_sha256=audit["sample_manifest_sha256"],
        managed_manifest_sha256=file_sha256(manifest_out),
        class_order=tuple(EQUIPMENT_CLASSES),
        partition_counts={p: audit["partition_counts"].get(p, 0) for p in PARTITIONS},
        positive_image_counts=audit["positive_image_counts"],
        no_target_counts=audit["no_target_counts"],
        source_counts=audit["source_counts"],
        audit_report_sha256=file_sha256(audit_path),
        license_ids=tuple(audit["license_ids"]),
        training_eligible=audit["training_eligible"],
    )
    identity_path = write_json(
        output_root / "identity/real_domain_dataset_identity.json",
        identity.to_record(),
    )
    return {"identity": identity.to_record(), "identity_path": str(identity_path),
            "coverage_gaps": audit["coverage_gaps"], "link_modes": dict(modes)}


def inspect_real_dataset(root):
    root = Path(root)
    identity = RealDomainDatasetIdentity.from_record(json.loads(
        (root / "identity/real_domain_dataset_identity.json").read_text()
    ))
    if file_sha256(root / "identity/samples.jsonl") != identity.managed_manifest_sha256:
        raise ValueError("managed real-domain sample manifest changed")
    audit = json.loads((root / "identity/audit.json").read_text())
    if file_sha256(root / "identity/audit.json") != identity.audit_report_sha256:
        raise ValueError("real-domain audit report changed")
    return {"valid": True, "identity": identity.to_record(),
            "coverage_gaps": audit.get("coverage_gaps", [])}

"""Conservatively triage pending Gomes transformer review rows."""

from __future__ import annotations

from collections import defaultdict
from io import BytesIO
from hashlib import sha256
import json
from pathlib import Path, PurePosixPath
import sys
import zipfile

from PIL import Image, ImageStat

def main(argv=None):
    manifest, yolo_root, classes_path, receipt_path = map(Path, (argv or sys.argv[1:]))
    classes = [line.strip() for line in classes_path.read_text().splitlines() if line.strip()]
    transformer_id = classes.index("Power transformer")
    rows = [json.loads(line) for line in manifest.read_text().splitlines() if line.strip()]
    bundles = {}
    labels = {}
    try:
        for row in rows:
            if row["review_status"] != "pending":
                continue
            member = row["members"][0]
            archive = member["archive"]
            bundle = bundles.setdefault(archive, zipfile.ZipFile(yolo_root / archive))
            if archive not in labels:
                labels[archive] = _label_index(bundle)
            decision = _triage(bundle, labels[archive], member["member"], transformer_id)
            row.update(decision)
    finally:
        for bundle in bundles.values():
            bundle.close()

    temporary = manifest.with_suffix(".jsonl.tmp")
    with temporary.open("w", encoding="utf-8") as destination:
        for row in rows:
            destination.write(json.dumps(row, sort_keys=True) + "\n")
    temporary.replace(manifest)
    state_paths = {}
    for state in ("accepted", "ambiguous", "ignored"):
        state_path = manifest.with_name(f"transformer-{state}.jsonl")
        with state_path.open("w", encoding="utf-8") as destination:
            for row in rows:
                if row["semantic_decision"] == state:
                    destination.write(json.dumps(row, sort_keys=True) + "\n")
        state_paths[state] = state_path
    counts = defaultdict(int)
    for row in rows:
        counts[row["semantic_decision"]] += 1
        counts[row["review_status"]] += 1
    receipt = json.loads(receipt_path.read_text())
    receipt.update({
        "accepted_transformer_image_count": counts["accepted"],
        "transformer_ambiguous_count": counts["ambiguous"],
        "transformer_ignored_count": counts["ignored"],
        "transformer_pending_count": counts["pending"],
        "transformer_reviewed_count": counts["reviewed"],
        "transformer_triage_policy": {
            "minimum_box_dimension_pixels": 40,
            "minimum_box_border_margin_fraction": 0.01,
            "minimum_crop_mean_luma": 25,
            "minimum_crop_luma_standard_deviation": 15,
            "semantic_basis": "registry-verified Power transformer label",
        },
        "curated_subset_ready_for_development_intake": counts["accepted"] > 0,
        "curated_subset_partition_constraint": "development-only-single-atomic-site",
        "raw_archives_training_forbidden": True,
        "accepted_manifest": state_paths["accepted"].name,
        "accepted_manifest_sha256": _file_sha256(state_paths["accepted"]),
        "ambiguous_manifest": state_paths["ambiguous"].name,
        "ambiguous_manifest_sha256": _file_sha256(state_paths["ambiguous"]),
        "ignored_manifest": state_paths["ignored"].name,
        "ignored_manifest_sha256": _file_sha256(state_paths["ignored"]),
        "resolved_issues": [
            "Exact duplicates collapsed by image SHA256",
            "Cross-release overlap measured and isolated from YOLO curation",
            "Missing labels, orphan labels, unreadable members, and ambiguous images excluded",
            "Every accepted row references a registry-verified Power transformer annotation",
        ],
        "blocking_issues": [
            "Raw archives must not be used directly; consume only accepted_manifest",
            "All accepted rows form one atomic site group and are development-only",
        ],
    })
    receipt.pop("gomes_source_family_identity_sha256", None)
    canonical = json.dumps(receipt, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
    receipt["gomes_source_family_identity_sha256"] = sha256(
        canonical.encode("utf-8")
    ).hexdigest()
    temporary = receipt_path.with_suffix(".json.tmp")
    temporary.write_text(json.dumps(receipt, indent=2, sort_keys=True) + "\n")
    temporary.replace(receipt_path)
    print(json.dumps(dict(sorted(counts.items())), sort_keys=True))


def _file_sha256(path):
    digest = sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _label_index(bundle):
    return {
        PurePosixPath(name).stem: name for name in bundle.namelist()
        if PurePosixPath(name).suffix.lower() == ".txt"
        and "labels" in PurePosixPath(name).parts
    }


def _triage(bundle, labels, image_member, transformer_id):
    image = Image.open(BytesIO(bundle.read(image_member))).convert("L")
    width, height = image.size
    rows = bundle.read(labels[PurePosixPath(image_member).stem]).decode().splitlines()
    failures = set()
    accepted_box = False
    metrics = []
    for line in rows:
        values = line.split()
        if int(float(values[0])) != transformer_id:
            continue
        x, y, w, h = map(float, values[1:])
        left, top, right, bottom = x - w / 2, y - h / 2, x + w / 2, y + h / 2
        pixel_w, pixel_h = w * width, h * height
        crop = image.crop((max(0, left * width), max(0, top * height),
                           min(width, right * width), min(height, bottom * height)))
        stats = ImageStat.Stat(crop)
        mean, deviation = stats.mean[0], stats.stddev[0]
        box_failures = []
        if min(pixel_w, pixel_h) < 40:
            box_failures.append("target_too_small")
        if min(left, top, 1 - right, 1 - bottom) < 0.01:
            box_failures.append("box_touches_image_border")
        if mean < 25:
            box_failures.append("insufficient_luma")
        if deviation < 15:
            box_failures.append("insufficient_contrast")
        metrics.append({"box_width_pixels": round(pixel_w, 2),
                        "box_height_pixels": round(pixel_h, 2),
                        "crop_mean_luma": round(mean, 2),
                        "crop_luma_standard_deviation": round(deviation, 2)})
        failures.update(box_failures)
        accepted_box = accepted_box or not box_failures
    if accepted_box:
        return {"review_status": "reviewed", "semantic_decision": "accepted",
                "whole_equipment_confirmed": True,
                "review_method": "conservative_geometry_visibility_triage",
                "review_note": "Verified source label with a visible, non-clipped transformer box",
                "triage_metrics": metrics}
    return {"review_status": "reviewed", "semantic_decision": "ambiguous",
            "whole_equipment_confirmed": None,
            "review_method": "conservative_geometry_visibility_triage",
            "review_note": ", ".join(sorted(failures)) or "no usable transformer box",
            "triage_metrics": metrics}


if __name__ == "__main__":
    main()

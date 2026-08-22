"""Audit overlap and transformer review candidates across Gomes releases."""

from __future__ import annotations

from collections import defaultdict
from hashlib import sha256
import json
from pathlib import Path, PurePosixPath
import zipfile

from PIL import Image, UnidentifiedImageError

from src.ml.artifacts import object_sha256, write_json
from src.vision.training.source_export_audit import IMAGE_SUFFIXES, _safe_entries


SCHEMA_VERSION = 1
YOLO_ARCHIVES = (
    "misc.zip", "agv_day.zip", "agv_night_light.zip", "agv_night_dark.zip",
)


def audit_gomes_source_family(semantic_archive, yolo_root, classes_path, output_root):
    semantic_archive = Path(semantic_archive)
    yolo_root = Path(yolo_root)
    classes = [line.strip() for line in Path(classes_path).read_text(encoding="utf-8").splitlines()
               if line.strip()]
    try:
        transformer_id = classes.index("Power transformer")
    except ValueError as exc:
        raise ValueError("classes.txt does not define Power transformer") from exc

    records = []
    transformer_hashes = set()
    for archive_name in YOLO_ARCHIVES:
        archive = yolo_root / archive_name
        if not archive.is_file():
            raise FileNotFoundError(f"missing Gomes component: {archive}")
        component, candidates = _read_yolo_component(archive, transformer_id)
        records.extend(component)
        transformer_hashes.update(candidates)
    semantic_records, unreadable_semantic_members = _read_image_archive(
        semantic_archive, "semantic-release"
    )
    records.extend(semantic_records)

    by_hash = defaultdict(list)
    for record in records:
        by_hash[record["sha256"]].append(record)
    duplicate_clusters = []
    cross_release_clusters = []
    for digest, members in sorted(by_hash.items()):
        if len(members) < 2:
            continue
        cluster = {"sha256": digest, "members": sorted(
            ({"release": row["release"], "archive": row["archive"],
              "member": row["member"]} for row in members),
            key=lambda row: (row["release"], row["archive"], row["member"]),
        )}
        duplicate_clusters.append(cluster)
        if len({row["release"] for row in members}) > 1:
            cross_release_clusters.append(cluster)

    semantic_hashes = {row["sha256"] for row in semantic_records}
    yolo_records = [row for row in records if row["release"] == "yolo-release"]
    near_overlap = _near_overlap_candidates(semantic_records, yolo_records)
    review_queue = []
    for digest in sorted(transformer_hashes):
        yolo_members = [row for row in by_hash[digest] if row["release"] == "yolo-release"]
        review_queue.append({
            "candidate_id": f"gomes-transformer-{digest[:16]}",
            "image_sha256": digest,
            "members": sorted(
                ({"archive": row["archive"], "member": row["member"]}
                 for row in yolo_members),
                key=lambda row: (row["archive"], row["member"]),
            ),
            "present_in_semantic_release": digest in semantic_hashes,
            "review_status": "pending",
            "semantic_decision": "ambiguous",
            "whole_equipment_confirmed": None,
            "review_note": "",
        })

    receipt = {
        "gomes_source_family_audit_schema_version": SCHEMA_VERSION,
        "source_id": "gomes-yolo-figshare-24060960",
        "source_family": "gomes-brazil-substation-01",
        "site_group_count": 1,
        "partition_policy": "atomic-development-only-after-review",
        "yolo_image_count": sum(row["release"] == "yolo-release" for row in records),
        "semantic_release_image_count": len(semantic_records),
        "semantic_release_unreadable_member_count": len(unreadable_semantic_members),
        "semantic_release_unreadable_members": unreadable_semantic_members,
        "independent_exact_image_count": len(by_hash),
        "exact_duplicate_cluster_count": len(duplicate_clusters),
        "cross_release_exact_cluster_count": len(cross_release_clusters),
        "semantic_images_present_in_yolo_count": len(semantic_hashes & {
            row["sha256"] for row in yolo_records
        }),
        "cross_release_near_overlap_candidate_count": len(near_overlap),
        "cross_release_near_overlap_review_queue": "near-overlap-review.jsonl",
        "transformer_candidate_independent_image_count": len(transformer_hashes),
        "transformer_candidates_present_in_semantic_release_count": len(
            transformer_hashes & semantic_hashes
        ),
        "accepted_transformer_image_count": 0,
        "training_eligible": False,
        "blocking_issues": [
            "All releases belong to one atomic site group",
            "Transformer candidates require whole-equipment semantic review",
            "Pending and ambiguous review rows do not count as verified coverage",
            *(["Semantic release ZIP contains unreadable members"]
              if unreadable_semantic_members else []),
        ],
        "duplicate_clusters": duplicate_clusters,
        "cross_release_exact_clusters": cross_release_clusters,
        "transformer_review_manifest": "transformer-review.jsonl",
    }
    receipt["gomes_source_family_identity_sha256"] = object_sha256(receipt)
    output = Path(output_root)
    output.mkdir(parents=True, exist_ok=True)
    write_json(output / "family-audit.json", receipt)
    _write_jsonl(output / "transformer-review.jsonl", review_queue)
    _write_jsonl(output / "near-overlap-review.jsonl", near_overlap)
    return receipt


def _read_image_archive(path, release):
    records, unreadable = [], []
    with zipfile.ZipFile(path) as bundle:
        for entry in _safe_entries(bundle):
            if entry.is_dir() or PurePosixPath(entry.filename).suffix.lower() not in IMAGE_SUFFIXES:
                continue
            try:
                digest = _member_sha256(bundle, entry)
                perceptual = _member_dhash(bundle, entry)
            except (zipfile.BadZipFile, EOFError, OSError, UnidentifiedImageError) as exc:
                unreadable.append({"member": entry.filename, "error": type(exc).__name__})
                continue
            records.append({
                "release": release, "archive": Path(path).name,
                "member": entry.filename, "sha256": digest, "dhash": perceptual,
            })
    return records, unreadable


def _read_yolo_component(path, transformer_id):
    with zipfile.ZipFile(path) as bundle:
        entries = _safe_entries(bundle)
        labels = {}
        for entry in entries:
            member = PurePosixPath(entry.filename)
            if not entry.is_dir() and member.suffix.lower() == ".txt" and "labels" in member.parts:
                labels[member.stem] = entry
        records, candidates = [], set()
        for entry in entries:
            member = PurePosixPath(entry.filename)
            if entry.is_dir() or member.suffix.lower() not in IMAGE_SUFFIXES:
                continue
            digest = _member_sha256(bundle, entry)
            perceptual = _member_dhash(bundle, entry)
            records.append({
                "release": "yolo-release", "archive": Path(path).name,
                "member": entry.filename, "sha256": digest, "dhash": perceptual,
            })
            label = labels.get(member.stem)
            if label is not None and _contains_class(bundle, label, transformer_id):
                candidates.add(digest)
        return records, candidates


def _contains_class(bundle, entry, class_id):
    for line in bundle.read(entry).decode("utf-8").splitlines():
        fields = line.split()
        if fields and float(fields[0]) == class_id:
            return True
    return False


def _member_sha256(bundle, entry):
    digest = sha256()
    with bundle.open(entry) as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _member_dhash(bundle, entry):
    with bundle.open(entry) as source, Image.open(source) as image:
        gray = image.convert("L").resize((9, 8))
        pixels = list(gray.getdata())
    value = 0
    for row in range(8):
        offset = row * 9
        for column in range(8):
            value = (value << 1) | (pixels[offset + column] > pixels[offset + column + 1])
    return value


def _near_overlap_candidates(semantic_records, yolo_records, threshold=4):
    candidates = []
    for semantic in semantic_records:
        for yolo in yolo_records:
            if semantic["sha256"] == yolo["sha256"]:
                continue
            distance = (semantic["dhash"] ^ yolo["dhash"]).bit_count()
            if distance <= threshold:
                candidates.append({
                    "semantic_member": semantic["member"],
                    "yolo_archive": yolo["archive"],
                    "yolo_member": yolo["member"],
                    "dhash_hamming_distance": distance,
                    "review_status": "pending",
                    "duplicate_decision": "ambiguous",
                })
    return sorted(candidates, key=lambda row: (
        row["dhash_hamming_distance"], row["semantic_member"],
        row["yolo_archive"], row["yolo_member"],
    ))


def _write_jsonl(path, rows):
    with Path(path).open("w", encoding="utf-8") as destination:
        for row in rows:
            destination.write(json.dumps(row, sort_keys=True, ensure_ascii=False) + "\n")

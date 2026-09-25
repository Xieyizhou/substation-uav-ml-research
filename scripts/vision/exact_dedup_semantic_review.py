#!/usr/bin/env python3
"""Run a SHA256-only deduplication pass over the reviewed hard-frame queue.

This command deliberately does not use perceptual hashes.  It binds every
candidate to its current image bytes, checks exact matches against the frozen
development/protected inventories and a prior candidate batch, and writes a
reviewable manifest.  It never changes an existing dataset and never releases
training data by itself.
"""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import json
from pathlib import Path
import re
import sys
from typing import Iterable

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.ml.artifacts import file_sha256, object_sha256, write_json


SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
DEFAULT_REVIEW = ROOT / "data/research/ml_training_recovery_v1/semantic-review-final.json"
DEFAULT_OUTPUT = ROOT / "data/research/ml_training_recovery_v1/exact-dedup-v1"


def _read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def _read_rows(path: Path, row_key: str | None = None) -> list[dict]:
    if path.suffix == ".jsonl":
        return _read_jsonl(path)
    value = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(value, list):
        return value
    if row_key is not None:
        rows = value.get(row_key)
    else:
        rows = value.get("selected") or value.get("kept") or value.get("members")
    if not isinstance(rows, list):
        raise ValueError(f"Reference does not contain a row array: {path}")
    return rows


def _resolve_source(row: dict, root: Path | None) -> Path | None:
    raw = row.get("image_path") or row.get("rgb_path") or row.get("image_relative_path")
    if not raw:
        return None
    path = Path(raw)
    if not path.is_absolute() and root is not None:
        path = root / path
    return path.resolve()


def _hash_from_row(row: dict) -> str | None:
    # The v2 membership calls the image payload hash payload_sha256; all other
    # inventories expose image_sha256 directly.
    return row.get("image_sha256") or row.get("payload_sha256")


def _short_row(row: dict, spec: dict, index: int) -> dict:
    image_sha256 = _hash_from_row(row)
    if not isinstance(image_sha256, str) or not SHA256_RE.fullmatch(image_sha256):
        raise ValueError(f"Invalid image hash in {spec['id']} row {index}")
    source = _resolve_source(row, spec.get("root"))
    result = {
        "reference_pool": spec["id"],
        "reference_role": spec["role"],
        "image_sha256": image_sha256,
    }
    if source is not None:
        result["source_path"] = str(source)
    for key in ("split", "sample_id", "frame_id", "source_frame_id", "recording_id", "collection_identity", "view_id", "map", "map_id"):
        if row.get(key) is not None:
            result[key] = row[key]
    return result


def _reference_specurations() -> list[dict]:
    dataset = ROOT / "data/research/visual_yolo_v2"
    canonical = ROOT / "data/research/canonical_views_v1"
    hard = ROOT / "data/research/visual_hard_examples_v2_12/run18-balanced-small-dense-hard-view-v1"
    return [
        {
            "id": "v2_development_replay",
            "role": "development",
            "path": dataset / "identity/train_membership.jsonl",
            "root": dataset,
        },
        {
            "id": "canonical_historical_kept",
            "role": "development",
            "path": canonical / "bounded-pilot-canonical-history-dedup-v1.json",
            "row_key": "kept",
        },
        {
            "id": "v2_protected_validation",
            "role": "protected",
            "path": canonical / "protected-v2-image-index-v1/images.jsonl",
            "root": dataset,
        },
        {
            "id": "protected_blind",
            "role": "protected",
            "path": canonical / "protected-blind-image-index-v1/images.jsonl",
            "root": dataset,
        },
        {
            "id": "protected_qualification",
            "role": "protected",
            "path": canonical / "protected-qualification-index-v1/images.jsonl",
        },
        {
            "id": "prior_run18_candidate_batch",
            "role": "candidate_batch",
            "path": hard / "membership.jsonl",
            "root": hard,
        },
        {
            "id": "canonical_expansion_round1_pending",
            "role": "candidate_batch",
            "path": canonical / "expansion-round1-joint-audit-v1.json",
            "row_key": "kept",
        },
    ]


def _load_references(specs: Iterable[dict]) -> tuple[dict[str, list[dict]], dict]:
    by_role: dict[str, list[dict]] = defaultdict(list)
    pool_report = {}
    for spec in specs:
        path = Path(spec["path"])
        if not path.exists():
            raise FileNotFoundError(path)
        rows = _read_rows(path, spec.get("row_key"))
        refs = [_short_row(row, spec, index) for index, row in enumerate(rows)]
        by_role[spec["role"]].extend(refs)
        hashes = [ref["image_sha256"] for ref in refs]
        pool_report[spec["id"]] = {
            "path": str(path),
            "path_sha256": file_sha256(path),
            "role": spec["role"],
            "row_count": len(refs),
            "unique_image_sha256_count": len(set(hashes)),
            "internal_duplicate_hash_count": sum(count > 1 for count in Counter(hashes).values()),
        }
    return by_role, pool_report


def _choose_reference(refs: list[dict]) -> dict:
    return sorted(
        refs,
        key=lambda ref: (
            ref.get("reference_pool", ""),
            ref.get("source_path", ""),
            ref.get("sample_id", ref.get("frame_id", ref.get("view_id", ""))),
        ),
    )[0]


def _candidate_record(row: dict, status: str, reason: str, matches: list[dict], duplicate_of: str | None = None) -> dict:
    result = {
        "frame_id": row["frame_id"],
        "map_id": row["map_id"],
        "collection_identity": row["collection_identity"],
        "image_path": row["image_path"],
        "image_sha256": row["image_sha256"],
        "source_truth_identity": row.get("source_truth_identity"),
        "semantic_decision": row["decision"],
        "target_status": row.get("target_status"),
        "training_admitted": False,
        "dedup_status": status,
        "dedup_reason": reason,
        "exact_duplicate_matches": matches,
    }
    if duplicate_of is not None:
        result["duplicate_of_frame_id"] = duplicate_of
    if row.get("taxonomy_confirmed") is not None:
        result["taxonomy_confirmed"] = row["taxonomy_confirmed"]
    if row.get("no_target_confirmed") is not None:
        result["no_target_confirmed"] = row["no_target_confirmed"]
    if row.get("removed_annotation_ids") is not None:
        result["removed_annotation_ids"] = row["removed_annotation_ids"]
    return result


def deduplicate(review_path: Path, output_dir: Path) -> dict:
    review = json.loads(review_path.read_text(encoding="utf-8"))
    rows = review.get("decisions")
    if not isinstance(rows, list) or not rows:
        raise ValueError("Semantic review has no decisions")
    if review.get("remaining_unreviewed_frames") != 0:
        raise ValueError("Exact dedup requires a closed semantic review")

    # The source review already rehashed every candidate.  Rehash again here
    # so the dedup artifact is independently bound to the bytes it selects.
    for row in rows:
        image = Path(row["image_path"])
        if not image.exists() or file_sha256(image) != row.get("image_sha256"):
            raise ValueError(f"Candidate image changed or is missing: {image}")
        if not SHA256_RE.fullmatch(row.get("image_sha256", "")):
            raise ValueError(f"Invalid candidate image hash: {row.get('image_sha256')}")

    specs = _reference_specurations()
    by_role, pool_report = _load_references(specs)
    role_hashes: dict[str, dict[str, list[dict]]] = {}
    for role, refs in by_role.items():
        role_hashes[role] = defaultdict(list)
        for ref in refs:
            role_hashes[role][ref["image_sha256"]].append(ref)

    # Track collisions in the reference inventories themselves.  This is a
    # finding about the existing pools, not a reason to rewrite them here.
    reference_cross_role = []
    all_reference_hashes = defaultdict(set)
    all_reference_rows = defaultdict(list)
    for role, refs in by_role.items():
        for ref in refs:
            all_reference_hashes[ref["image_sha256"]].add(role)
            all_reference_rows[ref["image_sha256"]].append(ref)
    for image_sha256, roles in sorted(all_reference_hashes.items()):
        if len(roles) > 1:
            rows_for_hash = all_reference_rows[image_sha256]
            reference_cross_role.append(
                {
                    "image_sha256": image_sha256,
                    "roles": sorted(roles),
                    "row_count": len(rows_for_hash),
                    "counts_by_pool": dict(Counter(row["reference_pool"] for row in rows_for_hash)),
                    "examples": [
                        {
                            key: row[key]
                            for key in ("reference_pool", "reference_role", "sample_id", "frame_id", "source_path")
                            if row.get(key) is not None
                        }
                        for row in sorted(rows_for_hash, key=lambda row: (row.get("reference_pool", ""), row.get("source_path", "")))[:8]
                    ],
                }
            )

    seen_candidates: dict[str, dict] = {}
    decisions = []
    selected = []
    reason_counts = Counter()
    semantic_counts = Counter()
    duplicate_matches_by_pool = Counter()
    candidate_internal_duplicates = []

    # Preserve semantic-review order in the output.  The winner of an
    # intra-batch collision is the first reviewed row, which is deterministic
    # because the review queue is identity-bound and ordered.
    for row in rows:
        semantic_counts[row["decision"]] += 1
        image_sha256 = row["image_sha256"]
        if row["decision"] != "accepted":
            reason = "semantic_review_excluded"
            reason_counts[reason] += 1
            decisions.append(_candidate_record(row, "excluded", reason, []))
            continue

        matches = []
        # Protected takes precedence over all development/candidate matches.
        for role in ("protected", "development", "candidate_batch"):
            for ref in role_hashes.get(role, {}).get(image_sha256, []):
                matches.append(ref)
                duplicate_matches_by_pool[ref["reference_pool"]] += 1
        if matches:
            primary = _choose_reference(matches)
            reason = {
                "protected": "exact_duplicate_protected",
                "development": "exact_duplicate_development",
                "candidate_batch": "exact_duplicate_candidate_batch",
            }[primary["reference_role"]]
            reason_counts[reason] += 1
            decisions.append(_candidate_record(row, "excluded", reason, matches))
            continue

        if image_sha256 in seen_candidates:
            winner = seen_candidates[image_sha256]
            match = {
                "reference_pool": "current_semantic_review_queue",
                "reference_role": "current_candidate",
                "image_sha256": image_sha256,
                "frame_id": winner["frame_id"],
                "source_path": winner["image_path"],
            }
            candidate_internal_duplicates.append({"image_sha256": image_sha256, "duplicate_frame_id": row["frame_id"], "kept_frame_id": winner["frame_id"]})
            reason_counts["exact_duplicate_within_candidate"] += 1
            decisions.append(_candidate_record(row, "excluded", "exact_duplicate_within_candidate", [match], winner["frame_id"]))
            continue

        seen_candidates[image_sha256] = row
        reason_counts["unique_exact_candidate"] += 1
        record = _candidate_record(row, "selected_for_next_gate", "unique_exact_candidate", [])
        decisions.append(record)
        selected.append(record)

    output_dir.mkdir(parents=True, exist_ok=True)
    decisions_path = output_dir / "decisions.jsonl"
    selected_path = output_dir / "selected.jsonl"
    cross_role_path = output_dir / "reference-cross-role-overlaps.jsonl"
    cross_role_rows = []
    for overlap in reference_cross_role:
        for ref in all_reference_rows[overlap["image_sha256"]]:
            cross_role_rows.append({"image_sha256": overlap["image_sha256"], **ref})
    cross_role_path.write_text(
        "".join(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n" for row in cross_role_rows),
        encoding="utf-8",
    )
    development_internal_path = output_dir / "development-internal-duplicates.jsonl"
    development_internal_rows = []
    for image_sha256, refs in sorted(role_hashes.get("development", {}).items()):
        if len(refs) > 1:
            for ref in refs:
                development_internal_rows.append({"image_sha256": image_sha256, **ref})
    development_internal_path.write_text(
        "".join(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n" for row in development_internal_rows),
        encoding="utf-8",
    )
    decisions_path.write_text(
        "".join(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n" for row in decisions),
        encoding="utf-8",
    )
    selected_path.write_text(
        "".join(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n" for row in selected),
        encoding="utf-8",
    )

    selected_semantic = [row for row in rows if row["decision"] == "accepted"]
    accepted_target = [row for row in selected_semantic if row.get("target_status") == "complete_taxonomy_targets_visible"]
    accepted_negative = [row for row in selected_semantic if row.get("target_status") == "no_taxonomy_target_visible"]
    selected_target = [row for row in selected if row.get("target_status") == "complete_taxonomy_targets_visible"]
    selected_negative = [row for row in selected if row.get("target_status") == "no_taxonomy_target_visible"]
    class_before = Counter(class_name for row in accepted_target for class_name in row.get("taxonomy_confirmed", []))
    class_after = Counter(class_name for row in selected_target for class_name in row.get("taxonomy_confirmed", []))

    result = {
        "schema_version": 1,
        "status": "exact_dedup_complete",
        "review_path": str(review_path),
        "review_sha256": file_sha256(review_path),
        "review_identity": review.get("review_identity"),
        "candidate_frame_count": len(rows),
        "semantic_decision_counts": dict(semantic_counts),
        "semantic_accepted_count": len(selected_semantic),
        "semantic_accepted_target_count": len(accepted_target),
        "semantic_accepted_negative_count": len(accepted_negative),
        "exact_dedup_reason_counts": dict(reason_counts),
        "selected_unique_exact_count": len(selected),
        "selected_unique_target_count": len(selected_target),
        "selected_unique_negative_count": len(selected_negative),
        "class_counts_before_exact_dedup": dict(class_before),
        "class_counts_after_exact_dedup": dict(class_after),
        "candidate_internal_duplicate_count": len(candidate_internal_duplicates),
        "candidate_internal_duplicates": candidate_internal_duplicates,
        "duplicate_matches_by_reference_pool": dict(duplicate_matches_by_pool),
        "reference_cross_role_overlap_count": len(reference_cross_role),
        "reference_cross_role_overlap_row_count": sum(item["row_count"] for item in reference_cross_role),
        "reference_cross_role_overlaps": reference_cross_role,
        "reference_cross_role_overlap_jsonl": str(cross_role_path),
        "reference_cross_role_overlap_jsonl_sha256": file_sha256(cross_role_path),
        "development_internal_duplicate_jsonl": str(development_internal_path),
        "development_internal_duplicate_jsonl_sha256": file_sha256(development_internal_path),
        "reference_pools": pool_report,
        "decisions_jsonl": str(decisions_path),
        "decisions_jsonl_sha256": file_sha256(decisions_path),
        "selected_jsonl": str(selected_path),
        "selected_jsonl_sha256": file_sha256(selected_path),
        "hash_basis": "Exact SHA256 of the source image bytes declared by each identity-bound membership/index; no perceptual or pixel-normalized matching was used.",
        "near_duplicate_performed": False,
        "training_admitted": False,
        "training_blockers": [
            "Exact dedup is a data hygiene gate, not training approval.",
            "Selected frames still require split/recording/view-lineage isolation and quota reconciliation.",
            "The 528 semantic framing exclusions remain quarantined and are not negative samples.",
            "Near-duplicate analysis is intentionally a separate next step.",
        ],
    }
    if reference_cross_role:
        result["training_blockers"].append(
            "Reference inventories contain an exact development/protected overlap; resolve the existing split contamination before formal training."
        )
    result["dedup_identity"] = object_sha256(result)
    write_json(output_dir / "report.json", result)
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--review", type=Path, default=DEFAULT_REVIEW)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    result = deduplicate(args.review, args.output)
    print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

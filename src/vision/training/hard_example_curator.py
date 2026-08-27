"""Deterministic, split-safe curation for Gazebo hard-example collections."""

from __future__ import annotations

from bisect import bisect_left
from collections import defaultdict
from dataclasses import dataclass, replace
import hashlib
import json
from pathlib import Path
from PIL import Image


def _canonical(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True)


def hamming_hex(left, right):
    return (int(left, 16) ^ int(right, 16)).bit_count()


def dhash64(path):
    with Image.open(path) as image:
        pixels = list(image.convert("L").resize((9, 8)).getdata())
    value = 0
    for row in range(8):
        offset = row * 9
        for column in range(8):
            value = (value << 1) | (pixels[offset + column] > pixels[offset + column + 1])
    return f"{value:016x}"


@dataclass(frozen=True)
class Candidate:
    collection: str
    collection_identity: str
    frame_id: str
    map_id: str
    split: str
    seed: int
    rgb_path: str
    rgb_timestamp: float
    image_sha256: str
    perceptual_hash: str
    objects: tuple[dict, ...]

    @property
    def kind(self):
        return "target" if self.objects else "no_target"

    @property
    def key(self):
        return self.collection, self.frame_id

    def record(self):
        return {
            "collection": self.collection,
            "collection_identity": self.collection_identity,
            "frame_id": self.frame_id,
            "image_sha256": self.image_sha256,
            "map_id": self.map_id,
            "objects": list(self.objects),
            "perceptual_hash": self.perceptual_hash,
            "rgb_path": self.rgb_path,
            "rgb_timestamp": self.rgb_timestamp,
            "seed": self.seed,
            "split": self.split,
        }


def _bucket(candidate):
    return ("all" if candidate.kind == "no_target" else candidate.map_id, candidate.kind, candidate.split)


def load_collection(collection, allowed_classes, maximum_truth_skew_ms=33.334, perceptual_hash_algorithm="receipt", hash_cache=None):
    collection = Path(collection).resolve()
    receipt = json.loads((collection / "collection-receipt.json").read_text())
    truth = [json.loads(line) for line in (collection / receipt["truth"]["relative_path"]).read_text().splitlines() if line.strip()]
    truth = sorted((row for row in truth if row.get("validation_status") == "valid"), key=lambda row: row["simulation_timestamp"])
    timestamps = [row["simulation_timestamp"] for row in truth]
    accepted, rejected = [], []
    hash_cache = {} if hash_cache is None else hash_cache
    for member in receipt["members"]:
        index = bisect_left(timestamps, member["rgb_timestamp"])
        indexes = [item for item in (index - 1, index) if 0 <= item < len(truth)]
        if not indexes:
            rejected.append({"frame_id": member["frame_id"], "reason": "missing_truth"})
            continue
        matched = min((truth[item] for item in indexes), key=lambda row: abs(row["simulation_timestamp"] - member["rgb_timestamp"]))
        skew_ms = abs(matched["simulation_timestamp"] - member["rgb_timestamp"]) * 1000.0
        if skew_ms > maximum_truth_skew_ms:
            rejected.append({"frame_id": member["frame_id"], "reason": "stale_truth", "skew_ms": skew_ms})
            continue
        objects = tuple(sorted((dict(item) for item in matched.get("objects", []) if item.get("validation_status") == "validated"), key=lambda item: item["annotation_id"]))
        unknown = sorted({item.get("class_name") for item in objects} - set(allowed_classes))
        if unknown:
            rejected.append({"frame_id": member["frame_id"], "reason": "unknown_truth_class", "classes": unknown})
            continue
        perceptual_hash = member["perceptual_hash"]
        if perceptual_hash_algorithm == "dhash64-v1":
            if member["image_sha256"] not in hash_cache:
                hash_cache[member["image_sha256"]] = dhash64(collection / member["rgb_path"])
            perceptual_hash = hash_cache[member["image_sha256"]]
        elif perceptual_hash_algorithm != "receipt":
            raise ValueError("unsupported perceptual hash algorithm")
        accepted.append(Candidate(
            str(collection), receipt["identity"], member["frame_id"], receipt["map_id"], receipt["split"],
            int(receipt["seed"]), member["rgb_path"], float(member["rgb_timestamp"]), member["image_sha256"],
            perceptual_hash, objects,
        ))
    return accepted, rejected, receipt["identity"]


def apply_bbox_policy(candidates, policy):
    """Reject frames whose labelled equipment violates split-specific framing rules."""
    if not policy:
        return list(candidates), []
    accepted, rejected = [], []
    for candidate in candidates:
        rules = policy.get(candidate.split, {})
        filter_objects = rules.get("mode") == "drop_invalid_objects"
        margin = float(rules.get("minimum_border_margin_px", 0.0))
        maximum_width = float(rules.get("maximum_bbox_width_fraction", 1.0)) * 1920.0
        maximum_height = float(rules.get("maximum_bbox_height_fraction", 1.0)) * 1080.0
        kept_objects = []
        dropped_reasons = []
        for item in candidate.objects:
            x1, y1, x2, y2 = map(float, item["bbox_xyxy"])
            if x1 <= margin or y1 <= margin or x2 >= 1920.0 - margin or y2 >= 1080.0 - margin:
                dropped_reasons.append("bbox_touches_frame_boundary")
            elif x2 - x1 > maximum_width or y2 - y1 > maximum_height:
                dropped_reasons.append("bbox_exceeds_framing_limit")
            else:
                kept_objects.append(item)
        if filter_objects and kept_objects:
            accepted.append(replace(candidate, objects=tuple(kept_objects)))
            rejected.extend({"frame_id": candidate.frame_id, "collection": candidate.collection, "reason": reason, "scope": "annotation"} for reason in dropped_reasons)
        elif not dropped_reasons:
            accepted.append(candidate)
        else:
            rejected.append({"frame_id": candidate.frame_id, "collection": candidate.collection, "reason": "no_complete_target_annotation" if filter_objects else dropped_reasons[0]})
    return accepted, rejected


def _clusters(candidates, threshold):
    clusters = []
    representatives = []
    for candidate in sorted(candidates, key=lambda row: (row.perceptual_hash, row.key)):
        cluster_index = next(
            (index for index, value in enumerate(representatives) if hamming_hex(candidate.perceptual_hash, value) <= threshold),
            None,
        )
        if cluster_index is None:
            representatives.append(candidate.perceptual_hash)
            clusters.append([candidate])
        else:
            clusters[cluster_index].append(candidate)
    return [sorted(rows, key=lambda row: row.key) for rows in clusters]


def curate(candidates, quotas, near_duplicate_hamming_threshold=6):
    rejected = []
    exact_groups = defaultdict(list)
    for candidate in candidates:
        exact_groups[candidate.image_sha256].append(candidate)
    exact_unique = []
    for sha256, rows in sorted(exact_groups.items()):
        rows.sort(key=lambda row: (row.split != "development", row.key))
        exact_unique.append(rows[0])
        rejected.extend({"frame_id": row.frame_id, "collection": row.collection, "reason": "exact_duplicate", "duplicate_of": rows[0].frame_id} for row in rows[1:])

    clusters = sorted(_clusters(sorted(exact_unique, key=lambda row: row.key), near_duplicate_hamming_threshold), key=lambda group: (-len(group), group[0].key))
    deficits = dict(quotas)
    assignments = []
    for rows in clusters:
        split_scores = {}
        for split in ("development", "validation"):
            counts = defaultdict(int)
            for row in rows:
                if row.split == split:
                    counts[_bucket(row)] += 1
            split_scores[split] = sum(
                min(count, deficits.get(bucket, 0))
                for bucket, count in counts.items()
            )
        assigned = max(("development", "validation"), key=lambda split: (split_scores[split], split == "validation"))
        assignments.append(assigned)
        for row in rows:
            if row.split != assigned:
                continue
            bucket = _bucket(row)
            deficits[bucket] = max(0, deficits.get(bucket, 0) - 1)

    def assignment_totals(values):
        totals = defaultdict(int)
        for rows, assigned in zip(clusters, values):
            for row in rows:
                if row.split == assigned:
                    totals[_bucket(row)] += 1
        return totals

    def objective(totals):
        return sum(min(totals.get(bucket, 0), required) for bucket, required in quotas.items())

    while True:
        totals = assignment_totals(assignments)
        baseline = objective(totals)
        best = None
        for index, assigned in enumerate(assignments):
            trial = list(assignments)
            trial[index] = "validation" if assigned == "development" else "development"
            gain = objective(assignment_totals(trial)) - baseline
            if gain > 0 and (best is None or (gain, -index) > (best[0], -best[1])):
                best = gain, index
        if best is None:
            break
        assignments[best[1]] = "validation" if assignments[best[1]] == "development" else "development"

    eligible = []
    cluster_records = []
    for cluster_id, (rows, assigned) in enumerate(zip(clusters, assignments), start=1):
        kept = [row for row in rows if row.split == assigned]
        eligible.extend(kept)
        for row in rows:
            if row.split != assigned:
                rejected.append({"frame_id": row.frame_id, "collection": row.collection, "reason": "near_duplicate_cross_split", "cluster_id": cluster_id, "assigned_split": assigned})
        cluster_records.append({"cluster_id": cluster_id, "assigned_split": assigned, "member_count": len(rows), "kept_count": len(kept), "perceptual_hashes": sorted({row.perceptual_hash for row in rows})})

    buckets = defaultdict(list)
    for candidate in eligible:
        buckets[_bucket(candidate)].append(candidate)
    selected = []
    coverage, shortfall = {}, {}
    for bucket, required in sorted(quotas.items()):
        rows = sorted(buckets.get(bucket, []), key=lambda row: (row.perceptual_hash, row.image_sha256, row.key))
        chosen = rows[:required]
        selected.extend(chosen)
        name = "/".join(bucket)
        coverage[name] = len(chosen)
        shortfall[name] = max(0, required - len(chosen))
        rejected.extend({"frame_id": row.frame_id, "collection": row.collection, "reason": "quota_excess"} for row in rows[required:])
    return sorted(selected, key=lambda row: (row.split, row.map_id, row.kind, row.key)), rejected, cluster_records, coverage, shortfall


def _multilabel_keys(candidate):
    if candidate.kind == "no_target":
        return (("all", "no_target", candidate.split),)
    return tuple(
        (class_name, "target", candidate.split)
        for class_name in sorted({item["class_name"] for item in candidate.objects})
    )


def curate_multilabel(candidates, quotas, near_duplicate_hamming_threshold=6):
    """Select split-isolated frames against per-class, multi-label quotas."""
    rejected = []
    exact_groups = defaultdict(list)
    for candidate in candidates:
        exact_groups[candidate.image_sha256].append(candidate)
    exact_unique = []
    for _, rows in sorted(exact_groups.items()):
        rows.sort(key=lambda row: (row.split != "development", row.key))
        exact_unique.append(rows[0])
        rejected.extend(
            {
                "frame_id": row.frame_id,
                "collection": row.collection,
                "reason": "exact_duplicate",
                "duplicate_of": rows[0].frame_id,
            }
            for row in rows[1:]
        )

    clusters = sorted(
        _clusters(
            sorted(exact_unique, key=lambda row: row.key),
            near_duplicate_hamming_threshold,
        ),
        key=lambda group: (-len(group), group[0].key),
    )
    assignments = []
    deficits = dict(quotas)
    for rows in clusters:
        split_scores = {}
        split_counts = {}
        for split in ("development", "validation"):
            counts = defaultdict(int)
            for row in rows:
                if row.split == split:
                    for key in _multilabel_keys(row):
                        counts[key] += 1
            split_counts[split] = counts
            split_scores[split] = sum(
                min(count, deficits.get(key, 0)) for key, count in counts.items()
            )
        assigned = max(
            ("development", "validation"),
            key=lambda split: (split_scores[split], split == "validation"),
        )
        assignments.append(assigned)
        for key, count in split_counts[assigned].items():
            deficits[key] = max(0, deficits.get(key, 0) - count)

    def assignment_totals(values):
        totals = defaultdict(int)
        for rows, assigned in zip(clusters, values):
            for row in rows:
                if row.split == assigned:
                    for key in _multilabel_keys(row):
                        totals[key] += 1
        return totals

    def objective(totals):
        return sum(
            min(totals.get(key, 0), required)
            for key, required in quotas.items()
        )

    while True:
        baseline = objective(assignment_totals(assignments))
        best = None
        for index, assigned in enumerate(assignments):
            trial = list(assignments)
            trial[index] = (
                "validation" if assigned == "development" else "development"
            )
            gain = objective(assignment_totals(trial)) - baseline
            if gain > 0 and (best is None or (gain, -index) > (best[0], -best[1])):
                best = gain, index
        if best is None:
            break
        assignments[best[1]] = (
            "validation"
            if assignments[best[1]] == "development"
            else "development"
        )

    eligible = []
    cluster_records = []
    for cluster_id, (rows, assigned) in enumerate(
        zip(clusters, assignments), start=1
    ):
        kept = [row for row in rows if row.split == assigned]
        eligible.extend(kept)
        for row in rows:
            if row.split != assigned:
                rejected.append(
                    {
                        "frame_id": row.frame_id,
                        "collection": row.collection,
                        "reason": "near_duplicate_cross_split",
                        "cluster_id": cluster_id,
                        "assigned_split": assigned,
                    }
                )
        cluster_records.append(
            {
                "cluster_id": cluster_id,
                "assigned_split": assigned,
                "member_count": len(rows),
                "kept_count": len(kept),
                "perceptual_hashes": sorted(
                    {row.perceptual_hash for row in rows}
                ),
            }
        )

    remaining = sorted(
        eligible,
        key=lambda row: (row.perceptual_hash, row.image_sha256, row.key),
    )
    deficits = dict(quotas)
    selected = []
    while any(value > 0 for value in deficits.values()):
        ranked = []
        for row in remaining:
            active = [key for key in _multilabel_keys(row) if deficits.get(key, 0) > 0]
            if not active:
                continue
            normalized_need = sum(
                deficits[key] / max(quotas[key], 1) for key in active
            )
            ranked.append(
                (-len(active), -normalized_need, row.perceptual_hash, row.image_sha256, row.key, row)
            )
        if not ranked:
            break
        chosen = min(ranked)[-1]
        selected.append(chosen)
        remaining.remove(chosen)
        for key in _multilabel_keys(chosen):
            deficits[key] = max(0, deficits.get(key, 0) - 1)

    selected_set = {row.key for row in selected}
    rejected.extend(
        {
            "frame_id": row.frame_id,
            "collection": row.collection,
            "reason": "quota_excess",
        }
        for row in remaining
        if row.key not in selected_set
    )
    totals = defaultdict(int)
    for row in selected:
        for key in _multilabel_keys(row):
            totals[key] += 1
    coverage = {"/".join(key): totals.get(key, 0) for key in sorted(quotas)}
    shortfall = {
        "/".join(key): max(0, required - totals.get(key, 0))
        for key, required in sorted(quotas.items())
    }
    return (
        sorted(selected, key=lambda row: (row.split, row.map_id, row.kind, row.key)),
        rejected,
        cluster_records,
        coverage,
        shortfall,
    )


def build_receipt(selected, rejected, clusters, coverage, shortfall, inputs, quotas, threshold, hash_algorithm):
    record = {
        "schema_version": 1,
        "status": "complete" if not any(shortfall.values()) else "shortfall",
        "input_collection_identities": sorted(inputs),
        "near_duplicate_hamming_threshold": threshold,
        "perceptual_hash_algorithm": hash_algorithm,
        "quotas": {"/".join(key): value for key, value in sorted(quotas.items())},
        "coverage": coverage,
        "shortfall": shortfall,
        "selected_count": len(selected),
        "rejected_count": len(rejected),
        "near_duplicate_cluster_count": len(clusters),
        "selected": [row.record() for row in selected],
    }
    record["identity"] = hashlib.sha256(_canonical(record).encode()).hexdigest()
    return record

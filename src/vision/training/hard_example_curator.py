"""Public curation API; input, framing and multi-label policies are separated."""

from collections import defaultdict
import hashlib
from src.vision.training.hard_example_candidates import (
    Candidate, _bucket, _canonical, _clusters, dhash64, hamming_hex, load_collection,
)
from src.vision.training.hard_example_framing import apply_bbox_policy
from src.vision.training.hard_example_multilabel import curate_multilabel, _multilabel_keys


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
        available_splits = tuple(
            split
            for split in ("development", "validation")
            if any(row.split == split for row in rows)
        )
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
        assigned = max(
            available_splits,
            key=lambda split: (split_scores[split], split == "validation"),
        )
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


def build_receipt(
    selected,
    rejected,
    clusters,
    coverage,
    shortfall,
    inputs,
    quotas,
    threshold,
    hash_algorithm,
    selection_group=None,
    deduplicate_within_split=False,
):
    record = {
        "schema_version": 1,
        "status": "complete" if not any(shortfall.values()) else "shortfall",
        "input_collection_identities": sorted(inputs),
        "near_duplicate_hamming_threshold": threshold,
        "perceptual_hash_algorithm": hash_algorithm,
        "selection_group": selection_group,
        "quotas": {"/".join(key): value for key, value in sorted(quotas.items())},
        "coverage": coverage,
        "shortfall": shortfall,
        "selected_count": len(selected),
        "rejected_count": len(rejected),
        "near_duplicate_cluster_count": len(clusters),
        "selected": [row.record() for row in selected],
    }
    if deduplicate_within_split:
        record["deduplicate_within_split"] = True
        record["selection_algorithm"] = "multilabel_pairwise_dhash_separated_v1"
        # Count distinct frames per class and size band, not annotation events.
        size_coverage = defaultdict(int)
        for row in selected:
            frame_keys = set()
            for item in row.objects:
                x1, y1, x2, y2 = map(float, item["bbox_xyxy"])
                area = max(0.0, x2 - x1) * max(0.0, y2 - y1) / (1920.0 * 1080.0)
                band = (
                    "lt_0_003" if area < 0.003 else
                    "0_003_to_0_01" if area < 0.01 else
                    "0_01_to_0_03" if area < 0.03 else "gte_0_03"
                )
                frame_keys.add((row.split, item["class_name"], band))
            for key in frame_keys:
                size_coverage["/".join(key)] += 1
        record["selected_class_size_frame_counts"] = dict(sorted(size_coverage.items()))
    record["identity"] = hashlib.sha256(_canonical(record).encode()).hexdigest()
    return record



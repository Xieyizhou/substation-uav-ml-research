"""Multi-label quota selection over indivisible duplicate clusters."""

from collections import defaultdict
from src.vision.training.hard_example_candidates import _clusters, hamming_hex


def _multilabel_keys(candidate):
    if candidate.kind == "no_target":
        return (("all", "no_target", candidate.split),)
    return tuple(
        (class_name, "target", candidate.split)
        for class_name in sorted({item["class_name"] for item in candidate.objects})
    )


def curate_multilabel(
    candidates,
    quotas,
    near_duplicate_hamming_threshold=6,
    selection_group=None,
    deduplicate_within_split=False,
):
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
        available_splits = tuple(
            split
            for split in ("development", "validation")
            if any(row.split == split for row in rows)
        )
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
            available_splits,
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
    selected_group_counts = defaultdict(int)
    while any(value > 0 for value in deficits.values()):
        ranked = []
        for row in remaining:
            active = [key for key in _multilabel_keys(row) if deficits.get(key, 0) > 0]
            if not active:
                continue
            normalized_need = sum(
                deficits[key] / max(quotas[key], 1) for key in active
            )
            group_load = 0
            if selection_group == "recording_seed":
                group_load = sum(
                    selected_group_counts[(key, row.seed)] for key in active
                )
            ranked.append(
                (
                    -len(active),
                    -normalized_need,
                    group_load,
                    row.perceptual_hash,
                    row.image_sha256,
                    row.key,
                    row,
                )
            )
        if not ranked:
            break
        chosen = min(ranked)[-1]
        selected.append(chosen)
        remaining.remove(chosen)
        if deduplicate_within_split:
            # Enforce pairwise separation from every selected frame, not just
            # split assignment against a cluster's initial representative.
            survivors = []
            for row in remaining:
                if hamming_hex(chosen.perceptual_hash, row.perceptual_hash) <= near_duplicate_hamming_threshold:
                    rejected.append({
                        "frame_id": row.frame_id,
                        "collection": row.collection,
                        "reason": (
                            "near_duplicate_within_split"
                            if row.split == chosen.split
                            else "near_duplicate_cross_split"
                        ),
                        "duplicate_of": chosen.frame_id,
                        "duplicate_collection": chosen.collection,
                    })
                else:
                    survivors.append(row)
            remaining = survivors
        for key in _multilabel_keys(chosen):
            deficits[key] = max(0, deficits.get(key, 0) - 1)
            if selection_group == "recording_seed":
                selected_group_counts[(key, chosen.seed)] += 1

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


#!/usr/bin/env python3
"""Create a deterministic split-safe manifest from successful hard-example flights."""

import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.vision.training.hard_example_curator import (
    apply_bbox_policy,
    build_receipt,
    curate,
    curate_multilabel,
    load_collection,
)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, action="append", required=True)
    parser.add_argument("--protocol", type=Path, default=ROOT / "config/perception/visual_hard_examples_v2_1.json")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    protocol = json.loads(args.protocol.read_text())
    candidates, rejected, identities = [], [], []
    hash_cache = {}
    seen_collections = set()
    for source in args.input:
        for run_receipt_path in sorted(source.resolve().glob("**/run-receipt.json")):
            run_receipt = json.loads(run_receipt_path.read_text())
            collection = run_receipt_path.parent / "collection"
            if run_receipt.get("status") != "complete" or not (collection / "collection-receipt.json").exists():
                continue
            if collection.resolve() in seen_collections:
                continue
            seen_collections.add(collection.resolve())
            rows, failures, identity = load_collection(collection, protocol["classes"], perceptual_hash_algorithm=protocol["perceptual_hash_algorithm"], hash_cache=hash_cache)
            rows, framing_failures = apply_bbox_policy(rows, protocol.get("bbox_policy"))
            candidates.extend(rows); rejected.extend(failures); identities.append(identity)
            rejected.extend(framing_failures)
    if "class_frame_quotas" in protocol:
        quotas = {
            (class_name, "target", split): int(count)
            for split, values in protocol["class_frame_quotas"].items()
            for class_name, count in values.items()
        }
        quotas.update(
            {
                ("all", "no_target", split): int(count)
                for split, count in protocol["no_target_frame_quotas"].items()
            }
        )
        selected, duplicate_rejections, clusters, coverage, shortfall = curate_multilabel(
            candidates,
            quotas,
            protocol["near_duplicate_hamming_threshold"],
            selection_group=protocol.get("selection_group"),
        )
    else:
        quotas = {
            **{(map_id, "target", split): count for map_id in protocol["maps"] for split, count in (("development", 600), ("validation", 300))},
            ("all", "no_target", "development"): 400,
            ("all", "no_target", "validation"): 200,
        }
        selected, duplicate_rejections, clusters, coverage, shortfall = curate(candidates, quotas, protocol["near_duplicate_hamming_threshold"])
    rejected.extend(duplicate_rejections)
    receipt = build_receipt(
        selected,
        rejected,
        clusters,
        coverage,
        shortfall,
        identities,
        quotas,
        protocol["near_duplicate_hamming_threshold"],
        protocol["perceptual_hash_algorithm"],
        selection_group=protocol.get("selection_group"),
    )
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / "curated-manifest.json").write_text(json.dumps(receipt, indent=2, sort_keys=True) + "\n")
    (args.output / "rejected.jsonl").write_text("".join(json.dumps(row, sort_keys=True) + "\n" for row in rejected))
    (args.output / "near-duplicate-clusters.json").write_text(json.dumps(clusters, indent=2, sort_keys=True) + "\n")
    print(json.dumps({key: receipt[key] for key in ("status", "identity", "selected_count", "coverage", "shortfall")}, indent=2, sort_keys=True))
    return 0 if receipt["status"] == "complete" else 2


if __name__ == "__main__":
    raise SystemExit(main())

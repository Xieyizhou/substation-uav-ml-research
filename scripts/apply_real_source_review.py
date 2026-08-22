"""Apply explicit visual-review policy to source review queues."""

from __future__ import annotations

from collections import Counter
from hashlib import sha256
import json
from pathlib import Path
import sys


def main(argv=None):
    review_root, policy_path = map(Path, (argv or sys.argv[1:]))
    policies = json.loads(policy_path.read_text())
    for source_id, source_policy in policies.items():
        source_root = review_root / source_id
        all_rows = []
        for review_path in sorted(source_root.glob("*/review.jsonl")):
            target = review_path.parent.name
            policy = source_policy if "default" in source_policy else source_policy[target]
            rows = [json.loads(line) for line in review_path.read_text().splitlines() if line]
            for index, row in enumerate(rows, start=1):
                decision = _decision(policy, index)
                row.update({
                    "review_status": "reviewed", "semantic_decision": decision,
                    "whole_equipment_confirmed": True if decision == "accepted" else None,
                    "review_method": "visual_contact_sheet_review",
                    "review_note": policy["reason"],
                })
                all_rows.append(row)
            _write_jsonl(review_path, rows)
        _write_source_receipt(source_root, source_id, all_rows)


def _decision(policy, index):
    if index in policy.get("accepted", []):
        return "accepted"
    if index in policy.get("ambiguous", []):
        return "ambiguous"
    if index in policy.get("ignored", []):
        return "ignored"
    return policy["default"]


def _write_source_receipt(root, source_id, rows):
    counts = Counter(row["semantic_decision"] for row in rows)
    accepted = Counter(row["target_class"] for row in rows
                       if row["semantic_decision"] == "accepted")
    for state in ("accepted", "ambiguous", "ignored"):
        _write_jsonl(root / f"semantic-{state}.jsonl",
                     [row for row in rows if row["semantic_decision"] == state])
    receipt = {
        "source_semantic_review_schema_version": 1,
        "source_id": source_id,
        "reviewed_image_class_rows": len(rows),
        "accepted_count": counts["accepted"],
        "ambiguous_count": counts["ambiguous"],
        "ignored_count": counts["ignored"],
        "pending_count": 0,
        "accepted_unique_image_count": dict(sorted(accepted.items())),
        "accepted_manifest": "semantic-accepted.jsonl",
        "ambiguous_manifest": "semantic-ambiguous.jsonl",
        "ignored_manifest": "semantic-ignored.jsonl",
        "raw_archive_training_forbidden": True,
    }
    canonical = json.dumps(receipt, sort_keys=True, separators=(",", ":"))
    receipt["source_semantic_review_identity_sha256"] = sha256(canonical.encode()).hexdigest()
    (root / "semantic-audit.json").write_text(
        json.dumps(receipt, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )


def _write_jsonl(path, rows):
    with path.open("w", encoding="utf-8") as destination:
        for row in rows:
            destination.write(json.dumps(row, sort_keys=True) + "\n")


if __name__ == "__main__":
    main()

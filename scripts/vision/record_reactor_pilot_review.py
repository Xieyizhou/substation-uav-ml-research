"""Freeze the explicit visual review of the source-isolated pilot frames."""
from datetime import datetime, timezone
from pathlib import Path
from scripts.vision.prepare_reactor_source_isolated import BASE, run as prepare
from src.ml.artifacts import file_sha256
from src.vision.canonical.plan import read_record, write_record


STATES = {
    "clear_near": ("clear_body", "近距离圆柱主体、顶面和基座完整可辨；前景未覆盖目标主体。"),
    "occluded": ("partial_occlusion", "目标主体存在且身份可解析，但前景块体覆盖部分下半部或侧面；保留遮挡状态。"),
    "edge_candidate": ("boundary_truncated", "目标框与图缘接触或极接近，仍有可辨主体；按图缘候选记录，不当作完整轮廓。"),
    "clear_far": ("small_clear", "远距离目标主体和基座可辨，目标较小；不见前景遮挡。"),
}


def run():
    plan = prepare()
    receipt_path = BASE / "pilot" / "collection-receipt.json"
    receipt = read_record(receipt_path)
    if receipt.get("status") != "complete_pending_review" or receipt.get("mode") != "pilot":
        raise ValueError("Pilot collection is not complete")
    expected = {x["view_id"]: x for x in plan["pilot_views"]}
    actual = {x["view_id"]: x for x in receipt["views"] if x.get("status") == "captured"}
    if set(actual) != set(expected) or len(actual) != 12:
        raise ValueError("Pilot population is incomplete or duplicated")
    now = datetime.now(timezone.utc).isoformat()
    decisions = []
    for view_id, view in expected.items():
        row = actual[view_id]
        checks = row.get("target_checks", {})
        if checks.get("planned_instance_present") is not True:
            raise ValueError(f"Planned instance absent: {view_id}")
        if row.get("expected_object_id") != view.get("object_id") or row.get("expected_category") != "reactor":
            raise ValueError(f"Pilot identity mismatch: {view_id}")
        state, reason = STATES[view["condition"]]
        view_json = BASE / "pilot" / f"{view_id}.json"
        rgb = Path(row["rgb_path"])
        depth = Path(row["depth_path"])
        paths = [BASE / "plan.json", receipt_path, view_json, rgb, depth]
        decisions.append({
            "view_id": view_id,
            "condition": view["condition"],
            "object_id": view["object_id"],
            "category": "reactor",
            "decision": "accepted_for_diagnostic_use",
            "technical_status": "captured_and_aligned",
            "visual_state": state,
            "reason": reason,
            "review_nature": "AI辅助审核",
            "reviewed_at": now,
            "pixel_visibility_certified": False,
            "target_checks": checks,
            "training_admitted": False,
            "promotable": False,
            "evidence_hashes": {str(p): file_sha256(p) for p in paths},
        })
    dest = BASE / "pilot-review.json"
    if dest.exists():
        return read_record(dest)
    return write_record(dest, {
        "schema_version": 1,
        "status": "pilot_review_complete_with_visibility_limits",
        "review_nature": "AI辅助审核",
        "reviewed_at": now,
        "plan_identity": plan["identity"],
        "collection_receipt": "pilot/collection-receipt.json",
        "collection_identity": receipt["identity"],
        "views": decisions,
        "note": "Diagnostic-only pilot review. Accepted means the planned instance is present and the frame is usable for diagnosis; it does not certify pixel-level visibility or training admission.",
        "training_admitted": False,
        "promotable": False,
        "inputs": {str(p): file_sha256(p) for p in [BASE / "plan.json", receipt_path, Path(__file__).resolve()]},
    })


if __name__ == "__main__":
    r = run()
    print(r["status"], r["identity"], len(r["views"]))

"""Freeze the human-facing calibration review required before pilot capture.

Calibration frames intentionally cover clear, occluded, edge and far views.  An
``accepted`` decision here means that the frame is technically usable as
calibration evidence and that the planned reactor instance is present; it does
not grant training admission or certify an area of visible pixels.
"""
from datetime import datetime, timezone
from pathlib import Path
from scripts.vision.prepare_reactor_source_isolated import BASE, run as prepare
from src.ml.artifacts import file_sha256
from src.vision.canonical.plan import read_record, write_record


STATES = {
    "clear_near": ("clear_body", "近距离圆柱主体、椭圆顶面和方形基座完整可辨；无明显前景覆盖。"),
    "occluded": ("partial_occlusion", "圆柱上部和顶面可辨；前景块体覆盖下部，保留为遮挡校准而非无遮挡训练证据。"),
    "edge_candidate": ("boundary_truncated", "目标框接触右图缘，主体仍有可辨结构；是否构成截断由正式逐框审核确认。"),
    "clear_far": ("small_clear", "远距离圆柱主体和基座可辨，目标较小但未见前景覆盖。"),
}


def run():
    plan = prepare()
    receipt_path = BASE / "calibration" / "collection-receipt.json"
    receipt = read_record(receipt_path)
    if receipt.get("status") != "complete_pending_review" or receipt.get("mode") != "calibration":
        raise ValueError("Calibration collection is not complete")
    expected = {x["view_id"]: x for x in plan["calibration_views"]}
    actual = {x["view_id"]: x for x in receipt["views"] if x.get("status") == "captured"}
    if set(actual) != set(expected) or len(actual) != 4:
        raise ValueError("Calibration population is incomplete or duplicated")
    now = datetime.now(timezone.utc).isoformat()
    decisions = []
    for view_id, view in expected.items():
        row = actual[view_id]
        if row.get("target_checks", {}).get("planned_instance_present") is not True:
            raise ValueError(f"Planned instance absent: {view_id}")
        if row.get("expected_object_id") != view.get("object_id") or row.get("expected_category") != "reactor":
            raise ValueError(f"Calibration identity mismatch: {view_id}")
        state, reason = STATES[view["condition"]]
        view_json = BASE / "calibration" / f"{view_id}.json"
        rgb = Path(row["rgb_path"])
        depth = Path(row["depth_path"])
        paths = [BASE / "plan.json", receipt_path, view_json, rgb, depth]
        decisions.append({
            "view_id": view_id,
            "condition": view["condition"],
            "object_id": view["object_id"],
            "category": "reactor",
            "decision": "accepted",
            "technical_status": "captured_and_aligned",
            "visual_state": state,
            "reason": reason,
            "review_nature": "AI辅助审核",
            "reviewed_at": now,
            "pixel_visibility_certified": False,
            "target_checks": row["target_checks"],
            "training_admitted": False,
            "promotable": False,
            "evidence_hashes": {str(p): file_sha256(p) for p in paths},
        })
    dest = BASE / "calibration-review.json"
    if dest.exists():
        return read_record(dest)
    return write_record(dest, {
        "schema_version": 1,
        "status": "accepted",
        "review_nature": "AI辅助审核",
        "reviewed_at": now,
        "plan_identity": plan["identity"],
        "collection_receipt": "calibration/collection-receipt.json",
        "collection_identity": receipt["identity"],
        "views": decisions,
        "note": "Calibration acceptance permits pilot capture only; it is not a training admission or pixel-level visibility certification.",
        "training_admitted": False,
        "promotable": False,
        "inputs": {str(p): file_sha256(p) for p in [BASE / "plan.json", receipt_path, Path(__file__).resolve()]},
    })


if __name__ == "__main__":
    r = run()
    print(r["status"], r["identity"], len(r["views"]))

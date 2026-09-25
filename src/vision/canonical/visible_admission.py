"""Versioned, offline-only adaptation of Gazebo visible pixel bounds."""

import copy
import hashlib
import json
import math


CLASSES = ("transformer", "switchgear", "capacitor_bank", "reactor")


def _identity(value):
    return hashlib.sha256(json.dumps(
        value, sort_keys=True, separators=(",", ":"), allow_nan=False
    ).encode("utf-8")).hexdigest()


def adapt_visible_truth(truth, *, annotation_mode):
    """Convert inclusive visible-pixel extrema; never infer full-object extent."""
    if annotation_mode != "visible_2d":
        raise ValueError("An explicit visible_2d source is required")
    if truth.get("validation_status") != "valid":
        raise ValueError("Invalid source truth")
    width, height = truth["image_width"], truth["image_height"]
    if any(type(v) is not int or v <= 0 for v in (width, height)):
        raise ValueError("Invalid image dimensions")
    source_identity = _identity(truth)
    result = copy.deepcopy(truth)
    seen = set()
    for item in result["objects"]:
        annotation_id = item["annotation_id"]
        if not isinstance(annotation_id, str) or not annotation_id or annotation_id in seen:
            raise ValueError("Invalid or duplicate annotation identity")
        seen.add(annotation_id)
        if item.get("class_name") not in CLASSES:
            raise ValueError("Unknown class")
        if item.get("class_id") != CLASSES.index(item["class_name"]):
            raise ValueError("Class identity mismatch")
        bounds = item["bbox_xyxy"]
        if len(bounds) != 4 or any(
            isinstance(v, bool) or not isinstance(v, (int, float))
            or not math.isfinite(v) or v != int(v) for v in bounds
        ):
            raise ValueError("Expected integer visible-pixel extrema")
        x1, y1, x2, y2 = map(int, bounds)
        if not (0 <= x1 <= x2 < width and 0 <= y1 <= y2 < height):
            raise ValueError("Visible bounds outside image")
        item["source_bbox_xyxy"] = list(bounds)
        item["source_bbox_coordinate_convention"] = "xyxy_pixel_extrema_inclusive"
        item["source_truncation_status"] = item.get("truncation_status")
        item["bbox_xyxy"] = [x1, y1, x2 + 1, y2 + 1]
        item["bbox_coordinate_convention"] = "xyxy_pixels_half_open"
        item["image_edge_contact"] = [name for name, contact in (
            ("left", x1 == 0), ("top", y1 == 0),
            ("right", x2 == width - 1), ("bottom", y2 == height - 1)
        ) if contact]
        # Visible extrema do not establish complete geometry or occlusion ratio.
        item["truncation_status"] = "unknown"
        item["visibility_status"] = "visible_extent_only"
        item["visible_fraction"] = None
        item["review_required"] = True
    result["adapter"] = "canonical-visible-extrema-v1"
    result["annotation_mode"] = annotation_mode
    result["source_truth_identity"] = source_identity
    result["training_admitted"] = False
    result["adapter_identity"] = _identity(result)
    return result


def review_frame(adapted, review, *, image_sha256):
    """Semantic review only; cannot grant deduplication or training admission."""
    expected = {item["annotation_id"] for item in adapted["objects"]}
    supplied = review.get("objects", [])
    ids = [item.get("annotation_id") for item in supplied]
    reasons = []
    if review.get("image_sha256") != image_sha256 or not image_sha256:
        reasons.append("image_identity_mismatch")
    if review.get("adapter_identity") != adapted.get("adapter_identity"):
        reasons.append("adapter_identity_mismatch")
    if len(ids) != len(set(ids)) or set(ids) != expected:
        reasons.append("incomplete_or_duplicate_object_review")
    if any(item.get("status") != "accepted" for item in supplied):
        reasons.append("unresolved_object")
    if any(item.get("visible_extent_confirmed") is not True for item in supplied):
        reasons.append("visible_extent_unconfirmed")
    if any(item.get("class_evidence_confirmed") is not True for item in supplied):
        reasons.append("class_evidence_unconfirmed")
    if any(item.get("framing_status") != "accepted" for item in supplied):
        reasons.append("framing_unconfirmed_or_rejected")
    if review.get("all_visible_targets_correct") is not True:
        reasons.append("annotation_completeness_unconfirmed")
    if not expected and review.get("no_target_confirmed") is not True:
        reasons.append("background_unconfirmed")
    return {
        "status": "quarantined" if reasons else "semantic_review_passed",
        "reasons": reasons,
        "retained_annotation_count": len(expected),
        "drop_annotations": False,
        "training_admitted": False,
        "remaining_gates": ["framing_policy", "split_isolation", "deduplication", "quota"],
    }

"""Validation for the versioned static visual benchmark templates."""

from __future__ import annotations

import json
from pathlib import Path

from src.ml import EQUIPMENT_CLASSES
from src.ml.artifacts import object_sha256
from src.ml.visual_pilot_acceptance import validate_v3_acceptance_policy
from src.ml.visual_identity import _required_text, class_order_identity


TEMPLATE_INFERENCE_POLICIES = frozenset({"every_frame", "every_nth_frame"})


def load_json(path):
    try:
        return json.loads(Path(path).read_text(encoding="utf-8"))
    except FileNotFoundError:
        raise
    except json.JSONDecodeError as error:
        raise ValueError(f"malformed JSON in {Path(path).name}: {error}") from error


def _template_identity(row):
    return object_sha256(
        {
            "condition_template_schema_version": 1,
            "input_size": row.get("input_size"),
            "inference_policy": row.get("inference_policy"),
            "frame_skip_interval": row.get("frame_skip_interval"),
            "target_inference_rate_hz": row.get("target_inference_rate_hz"),
            "roi_mode": row.get("roi_mode"),
            "batch_size": row.get("batch_size"),
            "warmup_frame_count": row.get("warmup_frame_count"),
            "measured_frame_count": row.get("measured_frame_count"),
        }
    )


def validate_static_benchmark_directory(benchmark_directory):
    root = Path(benchmark_directory)
    benchmark = load_json(root / "benchmark.json")
    conditions = load_json(root / "conditions.json")
    metrics = load_json(root / "metrics.json")
    pilot = load_json(root / "pilot_protocol.json")
    class_order = load_json(root / "class_order.json")
    if benchmark.get("benchmark_schema_version") != 1:
        raise ValueError("unsupported visual static benchmark schema")
    if benchmark.get("benchmark_id") != "visual_static_v1":
        raise ValueError("unexpected visual benchmark ID")
    if benchmark.get("materialization_status") != "template":
        raise ValueError("visual static v1 benchmark must remain a template")
    if metrics.get("metrics_schema_version") != 1:
        raise ValueError("unsupported visual metrics schema")
    if pilot.get("pilot_protocol_schema_version") != 1:
        raise ValueError("unsupported visual pilot protocol schema")
    if (
        pilot.get("protocol_version") != 3
        or pilot.get("protocol_id") != "visual-pilot-png-v3"
    ):
        raise ValueError("visual pilot must use the reviewed protocol v3")
    if pilot.get("canonical_payload_format") != "png":
        raise ValueError("visual pilot canonical payload must be png")
    recording = pilot.get("recording") or {}
    if (
        recording.get("expected_source_rate_hz") != 10.0
        or not recording.get("retain_every_valid_source_frame")
        or recording.get("automatic_downsampling") is not False
    ):
        raise ValueError("visual pilot v3 all-frame recording policy is invalid")
    validate_v3_acceptance_policy(pilot)
    classes = tuple(
        item.get("class_name") for item in class_order.get("classes", [])
    )
    class_ids = tuple(item.get("class_id") for item in class_order.get("classes", []))
    if classes != tuple(EQUIPMENT_CLASSES) or class_ids != tuple(range(4)):
        raise ValueError("visual benchmark class order does not match the locked order")
    if class_order.get("class_order_identity") != class_order_identity():
        raise ValueError("visual benchmark class-order identity mismatch")
    rows = conditions.get("conditions")
    if conditions.get("condition_template_schema_version") != 1 or not isinstance(
        rows, list
    ):
        raise ValueError("invalid condition template document")
    expected = {
        (size, policy, interval)
        for size in (320, 416, 640)
        for policy, interval in (
            ("every_frame", 1),
            ("every_nth_frame", 2),
            ("every_nth_frame", 3),
        )
    }
    observed = set()
    template_ids = set()
    template_identities = set()
    expected_fields = {
        "template_id",
        "materialization_status",
        "dataset_identity",
        "model_identity",
        "input_size",
        "inference_policy",
        "frame_skip_interval",
        "target_inference_rate_hz",
        "roi_mode",
        "batch_size",
        "warmup_frame_count",
        "measured_frame_count",
    }
    for row in rows:
        if set(row) != expected_fields:
            raise ValueError("condition template fields do not match schema v1")
        if row.get("materialization_status") != "template":
            raise ValueError("static matrix rows must remain unmaterialized templates")
        if (
            row.get("dataset_identity") is not None
            or row.get("model_identity") is not None
        ):
            raise ValueError("template conditions cannot claim dataset or model identity")
        if row.get("roi_mode") != "disabled":
            raise ValueError("static matrix v1 does not enable ROI")
        if row.get("batch_size") != 1:
            raise ValueError("static matrix v1 requires batch_size=1")
        if (
            row.get("target_inference_rate_hz") is not None
            or row.get("measured_frame_count") is not None
        ):
            raise ValueError("unmaterialized template runtime counts must be null")
        warmup = row.get("warmup_frame_count")
        if isinstance(warmup, bool) or not isinstance(warmup, int) or warmup < 0:
            raise ValueError("template warmup_frame_count must be non-negative")
        if row.get("inference_policy") not in TEMPLATE_INFERENCE_POLICIES:
            raise ValueError("static matrix contains an unsupported policy")
        observed.add(
            (
                row.get("input_size"),
                row.get("inference_policy"),
                row.get("frame_skip_interval"),
            )
        )
        template_id = _required_text(row.get("template_id"), "template_id")
        if template_id in template_ids:
            raise ValueError("duplicate condition template ID")
        template_ids.add(template_id)
        identity = _template_identity(row)
        if identity in template_identities:
            raise ValueError("duplicate condition template identity")
        template_identities.add(identity)
    if observed != expected or len(rows) != 9:
        raise ValueError("visual static v1 must contain exactly nine templates")
    return {
        "benchmark_id": benchmark["benchmark_id"],
        "materialization_status": benchmark.get("materialization_status"),
        "pilot_protocol_id": pilot.get("protocol_id"),
        "template_count": len(rows),
        "templates": rows,
    }

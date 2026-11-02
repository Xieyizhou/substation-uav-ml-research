#!/usr/bin/env python3
"""Check curated historical excerpts; optionally compare their original files.

The default check is self-contained and uses only the Python standard library.
It does not reproduce a simulator run, model inference, training, or label review.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path, PurePosixPath
import re
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
CASES = (
    "telemetry-event-loop", "bounded-shutdown-diagnostics",
    "shared-magnetic-model", "e21-model-experiment",
)


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def close(actual: float, expected: float, name: str) -> None:
    require(math.isfinite(actual) and math.isclose(actual, expected, abs_tol=1e-12),
            f"Inconsistent {name}: {actual} != {expected}")


def canonical(value: Any) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"),
                      ensure_ascii=False, allow_nan=False).encode("utf-8")


def relative_path(raw: str) -> PurePosixPath:
    path = PurePosixPath(raw)
    require(bool(raw) and not path.is_absolute() and ".." not in path.parts
            and "\\" not in raw and path.as_posix() == raw,
            f"Invalid source path: {raw}")
    return path


def json_pointer(value: Any, pointer: str) -> Any:
    require(pointer.startswith("/"), "Expected an absolute JSON pointer")
    for part in pointer[1:].split("/"):
        part = part.replace("~1", "/").replace("~0", "~")
        value = value[int(part)] if isinstance(value, list) else value[part]
    return value


def check_excerpt(path: Path, source_root: Path | None) -> tuple[dict, int]:
    document = json.loads(path.read_text(encoding="utf-8"))
    body = {key: value for key, value in document.items() if key != "identity"}
    require(document["identity"] == hashlib.sha256(canonical(body)).hexdigest(),
            f"Excerpt identity mismatch: {path.name}")
    require(document["case_excerpt_schema_version"] == 1, "Unknown excerpt schema")
    require(document["case_id"] == path.stem, "Case ID does not match filename")
    require(document["evidence_role"] == "historical_excerpt"
            and document["formal_evidence"] is False
            and document["qualifies_current_source"] is False,
            "Historical excerpt qualification flags changed")
    sources = {}
    original_json = {}
    for source in document["sources"]:
        raw = source["path"]
        rel = relative_path(raw)
        require(raw not in sources and source["bundled"] is False,
                "Duplicate source or incorrect availability declaration")
        require(re.fullmatch(r"[0-9a-f]{64}", source["sha256"]) is not None,
                "Invalid source SHA-256")
        require(type(source["bytes"]) is int and source["bytes"] > 0,
                "Invalid source size")
        sources[raw] = source
        if source_root is not None:
            original = source_root.joinpath(*rel.parts)
            require(original.is_file() and not original.is_symlink()
                    and original.resolve().is_relative_to(source_root.resolve()),
                    f"Original source unavailable or unsafe: {raw}")
            data = original.read_bytes()
            require(len(data) == source["bytes"]
                    and hashlib.sha256(data).hexdigest() == source["sha256"],
                    f"Original source hash mismatch: {raw}")
            if original.suffix == ".json":
                original_json[raw] = json.loads(data)
    values = {}
    for name, selection in document["selected"].items():
        require(selection["source"] in sources, f"Unknown source for {name}")
        require(selection["pointer"].startswith("/"), f"Invalid pointer for {name}")
        if source_root is not None:
            original_value = json_pointer(original_json[selection["source"]],
                                          selection["pointer"])
            require(canonical(original_value) == canonical(selection["value"]),
                    f"Excerpt differs from original selection: {name}")
        values[name] = selection["value"]
    return values, len(sources)


def check_queue(queue: dict) -> None:
    for key in ("produced", "accepted", "forwarded", "pending", "dropped"):
        require(type(queue[key]) is int and queue[key] >= 0, "Invalid queue count")
    require(queue["produced"] == queue["accepted"] + queue["dropped"],
            "Queue produced != accepted + dropped")
    require(queue["accepted"] == queue["forwarded"] + queue["pending"],
            "Queue accepted != forwarded + pending")


def check_model(model: dict) -> None:
    require(model["frame_count"] == 56, "Unexpected historical test membership")
    metrics = [model["pooled"], *model["layouts"].values()]
    for metric in metrics:
        close(metric["threshold"], model["threshold"], "threshold")
        for cls, counts in metric["per_class"].items():
            tp, fp, fn = (counts[key] for key in ("tp", "fp", "fn"))
            require(all(type(n) is int and n >= 0 for n in (tp, fp, fn)),
                    f"Invalid detection counts: {cls}")
            close(counts["precision"], tp / (tp + fp) if tp + fp else 0, cls + " precision")
            close(counts["recall"], tp / (tp + fn) if tp + fn else 0, cls + " recall")
            close(counts["f1"], 2 * tp / (2 * tp + fp + fn)
                  if 2 * tp + fp + fn else 0, cls + " F1")
        close(metric["macro_f1"], sum(c["f1"] for c in metric["per_class"].values())
              / len(metric["per_class"]), "macro F1")
    layouts = list(model["layouts"].values())
    pooled = model["pooled"]
    for cls, counts in pooled["per_class"].items():
        for key in ("tp", "fp", "fn"):
            require(counts[key] == sum(m["per_class"][cls][key] for m in layouts),
                    "Pooled detection count differs from layout sum")
    count = sum(m["small_object_count"] for m in layouts)
    require(count == pooled["small_object_count"] == 22, "Unexpected small-object count")
    close(pooled["small_object_recall"],
          sum(m["small_object_recall"] * m["small_object_count"] for m in layouts) / count,
          "pooled small-object recall")


def verify(source_root: Path | None = None) -> dict:
    excerpts = {}
    source_count = 0
    for case in CASES:
        values, count = check_excerpt(ROOT / "data/sample_outputs/case_studies" / (case + ".json"),
                                      source_root)
        excerpts[case] = values
        source_count += count
    telemetry = excerpts[CASES[0]]
    require(telemetry["before_max_gap_s"] > 2 > telemetry["after_max_gap_s"],
            "Telemetry gap comparison changed")
    require(telemetry["before_rows"] < telemetry["after_rows"]
            and telemetry["task_status"] == "failed"
            and telemetry["no_ARM_or_TAKEOFF_verified_from_actual_ULog"] is True
            and telemetry["normal_ARM_qualification"] is False
            and telemetry["logger_cadence_only"] is True,
            "Telemetry case boundaries changed")
    shutdown = excerpts[CASES[1]]
    for condition in ("before", "after"):
        check_queue(shutdown[condition + "_queue_terminal"])
        require(shutdown[condition + "_queue_capacity_per_instance"] == 4096
                and shutdown[condition + "_simulator_original_shutdown_grace_s"] == 6,
                "Shutdown budgets changed")
    require(shutdown["before_queue_terminal"]["fault"] is True
            and shutdown["after_queue_terminal"]["fault"] is False,
            "Old queue failure must remain visible")
    for index in range(3):
        native = shutdown[f"after_port_{index}_native_execution"]
        require(native["reaped"] is True and native["only_owned_child_signalled"] is True
                and 0 < native["original_timeout_s"] <= 2,
                "Bounded diagnostic ownership/budget changed")
    magnetic = excerpts[CASES[2]]
    close(magnetic["model_declination_difference_deg"],
          magnetic["px4_declination_deg"] - magnetic["gz_declination_deg"], "declination difference")
    original, shared = magnetic["conditions"]
    close(magnetic["original_mean_heading_offset_reduction_deg"],
          original["mean_original_control_heading_error_deg"]
          - shared["mean_original_control_heading_error_deg"], "heading reduction")
    for condition in magnetic["conditions"]:
        check_queue(condition["queue"])
        require(condition["original_ground_observations"] == 75
                and condition["no_ARM_or_TAKEOFF_commands"] is True
                and condition["actual_mission_status"] == "failed"
                and condition["normal_ARM_qualification"] is False,
                "Ground-only comparison boundaries changed")
    model = excerpts[CASES[3]]
    deltas = model["paired_seed_score_deltas"]
    close(model["paired_gates"]["mean"]["value"], sum(deltas) / len(deltas), "paired mean")
    require(model["paired_gates"]["wins"]["value"] == sum(delta > 0 for delta in deltas),
            "Paired seed win count mismatch")
    require(model["seed_27_individual_passed"] is False,
            "Individual seed failure must remain visible")
    require(model["identity"] == model["frozen_selection_identity"]
            and model["selected_model_sha256"] == model["candidate"]["model_sha256"],
            "Frozen model identity mismatch")
    close(model["threshold"], model["candidate"]["threshold"], "frozen threshold")
    for name in ("candidate", "current", "parent"):
        check_model(model[name])
    close(model["test_gates"]["pooled_macro_delta_current"]["value"],
          model["candidate"]["pooled"]["macro_f1"] - model["current"]["pooled"]["macro_f1"],
          "candidate/current macro delta")
    close(model["test_gates"]["pooled_macro_delta_parent"]["value"],
          model["candidate"]["pooled"]["macro_f1"] - model["parent"]["pooled"]["macro_f1"],
          "candidate/parent macro delta")
    return {"status": "passed", "cases_checked": len(CASES),
            "source_references_checked": source_count,
            "raw_sources_reopened": source_root is not None,
            "current_source_flight_qualified": False,
            "training_or_inference_rerun": False,
            "scope": "Excerpt integrity and displayed arithmetic; optional original file/selection comparison"}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-root", type=Path,
                        help="Optional archive root containing the unbundled historical originals")
    args = parser.parse_args()
    try:
        result = verify(args.source_root)
    except (ValueError, KeyError, TypeError, OSError, IndexError) as error:
        print(json.dumps({"status": "failed", "error": str(error)}))
        return 1
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

"""Freeze and preflight a bounded hard-negative rehearsal control.

The control starts from the completed source-isolated visibility schedule and
appends one deterministic exposure of each reviewed hard-negative member.
It is intentionally a new diagnostic family; historical protocols are read
and hash-checked but never rewritten.
"""
import json
import hashlib
from collections import Counter
from pathlib import Path
from types import SimpleNamespace
from contextlib import ExitStack
from unittest.mock import patch

from src.ml.artifacts import file_sha256, object_sha256
from src.vision.canonical.plan import read_record, write_record

ROOT = Path(__file__).resolve().parents[2]
SOURCE = ROOT / "data/research/ml_training_recovery_v1/reactor-visibility-expansion-v4/training-control-v1"
OUT = SOURCE.parent / "negative-rehearsal-control-v2"
NEGATIVE_REVIEW = ROOT / "data/research/ml_training_recovery_v1/hard-negative-isolated-v2/semantic-review.json"
EVAL = SOURCE / "evaluation-v2"
SEEDS = (7, 17, 27)
KEYS = tuple(f"negative-rehearsal-480-{s}" for s in SEEDS)
NAMES = ("transformer", "switchgear", "capacitor_bank", "reactor")
BATCH = 6
STEPS = 480
BASE_DRAWS = 2760
EXTRA_DRAWS = 120


def _digest(*parts):
    return hashlib.sha256("|".join(map(str, ("negative-rehearsal-control-v1", *parts))).encode()).hexdigest()


def _audit():
    path = OUT / "fp-audit.json"
    if path.exists():
        return read_record(path)
    review = read_record(NEGATIVE_REVIEW)
    if review.get("status") != "reviewed" or review.get("accepted") != 48 or review.get("held"):
        raise ValueError("Negative review is not complete")
    by_key = {(r["view_id"], r.get("variant")): r for r in review["frames"]}
    events = []
    for seed in SEEDS:
        ep = EVAL / f"visibility-452-{seed}.json"
        if not ep.exists():
            raise ValueError(f"Missing completed evaluation for seed {seed}")
        record = read_record(ep)
        for row in record["negative_rows"]:
            key = (row["view_id"], row.get("variant"))
            source = by_key.get(key)
            if source is None:
                raise ValueError(f"Negative review identity missing: {key}")
            for index, pred in enumerate(row["predictions"]):
                subject = source.get("subject", "unknown")
                if "cabinet" in subject:
                    structure = "cabinet_body"
                elif "pole" in subject or "tower" in subject:
                    structure = "pole_or_crossarm"
                elif "building" in subject or "wall" in subject:
                    structure = "building_or_wall"
                else:
                    structure = "unknown"
                events.append({"seed": seed, "view_id": row["view_id"], "variant": row.get("variant"),
                               "prediction_index": index, "class_name": pred["class_name"],
                               "confidence": pred["confidence"], "bbox_xyxy": pred["bbox_xyxy"],
                               "image_sha256": row["image_sha256"], "source_subject": subject,
                               "structure_category": structure})
    grouped = Counter((e["view_id"], e["variant"]) for e in events)
    for e in events:
        e["same_image_seed_count"] = len({x["seed"] for x in events if (x["view_id"], x["variant"]) == (e["view_id"], e["variant"])})
        e["same_image_prediction_events"] = grouped[e["view_id"], e["variant"]]
    record = {"status": "technical_fp_audit_complete", "prediction_events": events,
              "unique_error_images": len({e["image_sha256"] for e in events}),
              "source_review_identity": review["identity"],
              "review_nature": "AI-assisted source review is reused as provenance; this file adds no semantic approval.",
              "training_admitted": False, "promotable": False,
              "inputs": {str(NEGATIVE_REVIEW): file_sha256(NEGATIVE_REVIEW),
                         **{str(EVAL / f"visibility-452-{s}.json"): file_sha256(EVAL / f"visibility-452-{s}.json") for s in SEEDS}}}
    return write_record(path, record)


def _schedule(rows, source_schedule, seed):
    hard = sorted((r["member_id"] for r in rows if r.get("subset") == "hard_negative"),
                  key=lambda mid: _digest(seed, mid, "hard-negative-order"))
    if len(hard) != EXTRA_DRAWS or len(source_schedule) != BASE_DRAWS:
        raise ValueError("Hard-negative or source schedule population changed")
    return list(source_schedule) + hard


def _exposure(rows, seq):
    by = {r["member_id"]: r for r in rows}; images = Counter(seq); classes = Counter(); subsets = Counter(); lineages = Counter()
    for mid, count in images.items():
        row = by[mid]; subsets[row["subset"]] += count; lineages[row["lineage_id"]] += count
        for cls, n in row.get("class_instances", {}).items(): classes[cls] += count * int(n)
    return {"draws": len(seq), "optimizer_steps": len(seq) // BATCH,
            "image_exposure": dict(images), "class_instance_exposure": dict(classes),
            "subset_exposure": dict(subsets), "lineage_exposure": dict(lineages)}


def freeze():
    OUT.mkdir(parents=True, exist_ok=True)
    path = OUT / "protocol.json"
    if path.exists():
        return read_record(path)
    source = read_record(SOURCE / "protocol.json")
    audit = _audit()
    rows = source["pool_rows"]
    if len(rows) != 388 or len({r["member_id"] for r in rows}) != 388:
        raise ValueError("Source pool membership changed")
    for row in rows:
        if file_sha256(row["image_path"]) != row["image_sha256"] or file_sha256(row["label_path"]) != row["label_sha256"]:
            raise ValueError(f"Source member hash changed: {row['member_id']}")
    schedules = {}; exposures = {}; listings = {}; deps = [SOURCE / "protocol.json", SOURCE / "entry-ready.json", OUT / "fp-audit.json", NEGATIVE_REVIEW, Path(__file__).resolve()]
    for seed in SEEDS:
        key = f"visibility-452-{seed}"; new_key = f"negative-rehearsal-480-{seed}"
        seq = _schedule(rows, source["schedules"][key], seed); schedules[new_key] = seq; exposures[new_key] = _exposure(rows, seq)
        listing = OUT / f"{new_key}.txt"; text = "".join(f"{next(r['image_path'] for r in rows if r['member_id'] == m)}\n" for m in sorted(set(seq)))
        if listing.exists() and listing.read_text() != text: raise ValueError("Listing drift")
        if not listing.exists(): listing.write_text(text)
        listings[new_key] = str(listing); deps.append(listing)
    protocol = {"schema_version": 1, "status": "frozen_negative_rehearsal_preflight_pending",
                "source_protocol": str(SOURCE / "protocol.json"), "source_protocol_identity": source["identity"],
                "pool_rows": rows, "schedules": schedules, "exposures": exposures, "listings": listings,
                "names": list(NAMES), "seeds": list(SEEDS), "steps": STEPS, "batch": BATCH,
                "initialization": source["initialization"],
                "training_config": {"device": "cpu", "imgsz": 640, "optimizer": "AdamW", "lr0": .0005,
                                    "warmup_epochs": 0, "enhancement": "all_closed", "initialization": "independent_v2.11"},
                "evaluation": {"device": "cpu", "imgsz": 640, "confidence": [.37, .001], "nms_iou": .7,
                               "max_det": 300, "matching_iou": .5},
                "comparison": "Retain the completed visibility schedule and append one deterministic exposure of all 120 reviewed hard-negative members.",
                "interpretation": "Bounded negative-rehearsal diagnostic; total exposure increases from 2760 to 2880 and is not a fixed-budget causal comparison.",
                "error_audit": str(OUT / "fp-audit.json"), "training_started": False,
                "training_admitted": False, "promotable": False, "unseen_scene_status": "sealed_not_evaluated",
                "inputs": {str(d): file_sha256(d) for d in deps}}
    return write_record(path, protocol)


def _preflight_cell(protocol, key):
    from scripts.vision import train_reactor_visibility_expansion as base
    import torch
    from ultralytics.data.utils import check_det_dataset
    from scripts.vision.order_retention_runtime import validate_labels
    base.OUT = OUT; base.STEPS = STEPS; base.KEYS = KEYS
    owner = SimpleNamespace(epoch=0); actual = []; batches = []
    lookup = {r["image_path"]: r["member_id"] for r in protocol["pool_rows"]}
    with ExitStack() as stack:
        stack.enter_context(patch.object(torch.optim.Optimizer, "__init__", side_effect=AssertionError("optimizer forbidden")))
        stack.enter_context(patch.object(torch.Tensor, "backward", side_effect=AssertionError("backward forbidden")))
        dataset, loader = base._loader(protocol, key, owner)
        validate_labels(dataset, {r["member_id"]: r for r in protocol["pool_rows"]})
        for epoch in range(STEPS // 10):
            owner.epoch = epoch
            for batch in loader:
                if tuple(batch["img"].shape) != (BATCH, 3, 640, 640): raise ValueError("Tensor shape drift")
                mids = [lookup[x] for x in batch["im_file"]]; pos = len(actual)
                if mids != protocol["schedules"][key][pos:pos + BATCH]: raise ValueError("Exposure order drift")
                actual.extend(mids); batches.append({"batch": len(batches), "members": mids, "label_count": int(batch["cls"].shape[0])})
    if actual != protocol["schedules"][key] or len(batches) != STEPS:
        raise ValueError("Preflight endpoint mismatch")
    return {"status": "actual_complete_loader_verified_no_training", "key": key,
            "actual": actual, "checked_draws": len(actual), "checked_batches": len(batches),
            "batch_records": batches, "optimizer_created": False, "backward_executed": False,
            "validation_run": False, "inputs": {str(OUT / "protocol.json"): file_sha256(OUT / "protocol.json"), str(Path(__file__).resolve()): file_sha256(Path(__file__).resolve())}}


def preflight():
    protocol = freeze(); root = OUT / "actual-preflight"; root.mkdir(exist_ok=True); records = []
    for key in KEYS:
        path = root / f"{key}.json"
        if path.exists(): record = read_record(path)
        else: record = write_record(path, _preflight_cell(protocol, key))
        if record["actual"] != protocol["schedules"][key] or record["checked_draws"] != STEPS * BATCH:
            raise ValueError(f"Invalid preflight: {key}")
        records.append(record); print("PREFLIGHT_COMPLETE", key, record["checked_draws"], flush=True)
    ready = OUT / "entry-ready.json"
    if not ready.exists():
        write_record(ready, {"status": "ready_for_training_not_started", "cells": list(KEYS),
                             "actual_draws_verified": sum(len(r["actual"]) for r in records),
                             "optimizer_created": False, "backward_executed": False,
                             "training_admitted": False, "promotable": False,
                             "inputs": {str(OUT / "protocol.json"): file_sha256(OUT / "protocol.json"),
                                        **{str(root / f"{k}.json"): file_sha256(root / f"{k}.json") for k in KEYS}}})
    r = read_record(ready)
    if r["status"] != "ready_for_training_not_started" or r["actual_draws_verified"] != STEPS * BATCH * len(KEYS):
        raise ValueError("Training readiness invalid")
    return r


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser(); ap.add_argument("--freeze", action="store_true"); ap.add_argument("--preflight", action="store_true"); args = ap.parse_args()
    if args.preflight: print(preflight()["status"])
    else: print(freeze()["status"])

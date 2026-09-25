"""Freeze and preflight a small source-isolated visibility expansion dataset.

This module deliberately creates a new development-only dataset.  It never
edits the reviewed pool and its preflight forbids optimizer/backward paths.
"""
from collections import Counter
from contextlib import ExitStack
from pathlib import Path
from types import SimpleNamespace
import copy
import hashlib
import json

from PIL import Image, ImageChops

from src.ml.artifacts import file_sha256, object_sha256
from src.vision.canonical.plan import read_record, write_record
from scripts.vision.prepare_reactor_source_isolated import BASE as COLLECTION_BASE, run as prepare_collection

ROOT = Path(__file__).resolve().parents[2]
OUT = COLLECTION_BASE / "training-control-v1"
OLD_MANIFEST = (ROOT / "data/research/ml_training_recovery_v1/source-isolated-material-retention-v1/closed-exterior-coverage-v1/source-retention-control-v1/repeated-budget-control-v1/cool-light-coverage-v1/neutral-gray-calibration-v1/small-scale-material-control-v1/interleaved-tail-control-v1/reviewed-dataset-fit-v1/reviewed-dataset-v1/manifest.json")
WEIGHTS = ROOT / "models/equipment/visual-yolo11n-baseline-v2.11-candidate-package-v1/weights/best.pt"
NAMES = ["transformer", "switchgear", "capacitor_bank", "reactor"]
SEEDS = (7, 17, 27)
STEPS = 460
BATCH = 6
BASE_DRAWS = 2700
EXTRA_DRAWS = 60


def _digest(*parts):
    return hashlib.sha256("|".join(map(str, ("reactor-visibility-expansion-v4", *parts))).encode()).hexdigest()


def _label_text(truth):
    w, h = 1920.0, 1080.0
    out = []
    for obj in truth["objects"]:
        cid = int(obj["class_id"])
        if not (0 <= cid < 4) or obj["class_name"] != NAMES[cid]:
            raise ValueError("Unexpected class mapping")
        x0, y0, x1, y1 = map(float, obj["bbox_xyxy"])
        if not (0 <= x0 < x1 <= w and 0 <= y0 < y1 <= h):
            raise ValueError("Out-of-image truth box")
        out.append(f"{cid} {(x0+x1)/(2*w):.12f} {(y0+y1)/(2*h):.12f} {(x1-x0)/w:.12f} {(y1-y0)/h:.12f}")
    return "\n".join(out) + ("\n" if out else "")


def _verify_evidence(review):
    for d in review["views"]:
        if d["decision"] != "accepted_for_diagnostic_use":
            raise ValueError("Pilot review is not complete")
        for path, digest in d["evidence_hashes"].items():
            if file_sha256(path) != digest:
                raise ValueError("Pilot evidence hash changed")


def _new_members(plan, receipt, review):
    by_id = {v["view_id"]: v for v in receipt["views"]}
    decisions = {v["view_id"]: v for v in review["views"]}
    rows = []
    image_dir, label_dir = OUT / "dataset/images", OUT / "dataset/labels"
    image_dir.mkdir(parents=True, exist_ok=True); label_dir.mkdir(parents=True, exist_ok=True)
    for view in plan["pilot_views"]:
        vid = view["view_id"]
        r, d = by_id[vid], decisions[vid]
        if r.get("status") != "captured" or r["target_checks"].get("planned_instance_present") is not True:
            raise ValueError("Pilot frame is not trainable evidence")
        if d["decision"] != "accepted_for_diagnostic_use":
            raise ValueError("Pilot review decision does not permit candidate export")
        src = Path(r["rgb_path"]); out_img = image_dir / f"extreme-{vid}.png"; out_lbl = label_dir / f"extreme-{vid}.txt"
        with Image.open(src) as im:
            rgb = im.convert("RGB")
            if rgb.size != (1920, 1080): raise ValueError("Unexpected pilot image dimensions")
            if out_img.exists():
                with Image.open(out_img) as old:
                    if old.size != rgb.size or ImageChops.difference(old.convert("RGB"), rgb).getbbox() is not None:
                        raise ValueError("Existing exported pixels differ")
            else: rgb.save(out_img, format="PNG")
        text = _label_text(r["truth"])
        if out_lbl.exists() and out_lbl.read_text() != text: raise ValueError("Existing exported labels differ")
        if not out_lbl.exists(): out_lbl.write_text(text)
        if file_sha256(src) != r["image_sha256"]: raise ValueError("Pilot RGB hash changed")
        if file_sha256(out_lbl) != file_sha256(out_lbl): raise ValueError("Label write failure")
        object_ids = Counter(o["class_name"] for o in r["truth"]["objects"])
        rows.append({
            "member_id": f"extreme:{vid}", "subset": "source_isolated_visibility", "variant": view["condition"],
            "image_path": str(out_img), "label_path": str(out_lbl),
            "image_sha256": file_sha256(out_img), "label_sha256": file_sha256(out_lbl),
            "source_image_path": str(src), "source_image_sha256": r["image_sha256"],
            "source_label_identity": object_sha256(r["truth"]), "lineage_id": f"extreme:{view['object_id']}:{view['condition']}",
            "pair_id": vid, "class_instances": dict(object_ids), "truth": r["truth"],
            "asset_family": "canonical-primitive-equipment", "data_role": "bounded_development_training_candidate",
            "visual_state": d["visual_state"], "review_identity": review["identity"],
            "training_admitted": False, "promotable": False,
        })
    if len(rows) != 12 or len({r["member_id"] for r in rows}) != 12: raise ValueError("Pilot member population changed")
    return rows


def _schedule(old_rows, new_rows, seed):
    old = sorted((r["member_id"] for r in old_rows), key=lambda m: _digest(seed, m, "base-order"))
    seq = (old * ((BASE_DRAWS + len(old) - 1) // len(old)))[:BASE_DRAWS]
    extras = sorted((r["member_id"] for r in new_rows), key=lambda m: _digest(seed, m, "extra-order"))
    seq.extend(extras * (EXTRA_DRAWS // len(extras)))
    if len(seq) != STEPS * BATCH: raise ValueError("Schedule length mismatch")
    return seq


def _exposure(rows, seq):
    by = {r["member_id"]: r for r in rows}
    images = Counter(seq); instances = Counter(); lineages = Counter(); subsets = Counter()
    for mid, n in images.items():
        r = by[mid]; subsets[r["subset"]] += n; lineages[r["lineage_id"]] += n
        for cls, count in r.get("class_instances", {}).items(): instances[cls] += n * int(count)
    return {"image_exposure": dict(images), "class_instance_exposure": dict(instances), "lineage_exposure": dict(lineages), "subset_exposure": dict(subsets), "draws": len(seq), "optimizer_steps": len(seq)//BATCH}


def freeze():
    path = OUT / "protocol.json"
    if path.exists(): return read_record(path)
    plan = prepare_collection(); pilot_receipt = read_record(COLLECTION_BASE / "pilot/collection-receipt.json")
    pilot_review = read_record(COLLECTION_BASE / "pilot-review.json"); _verify_evidence(pilot_review)
    if pilot_review["plan_identity"] != plan["identity"] or pilot_review["collection_identity"] != pilot_receipt["identity"]:
        raise ValueError("Pilot review identity mismatch")
    old = read_record(OLD_MANIFEST)
    old_rows = []
    for row in old["members"]:
        row = copy.deepcopy(row)
        for field in ("image_path", "label_path"):
            if file_sha256(row[field]) != row[field.replace("_path", "_sha256")]: raise ValueError("Historical member hash drift")
        row.setdefault("training_admitted", False); row.setdefault("promotable", False); old_rows.append(row)
    if len(old_rows) != 376 or len({r["member_id"] for r in old_rows}) != 376: raise ValueError("Reviewed pool population changed")
    new_rows = _new_members(plan, pilot_receipt, pilot_review); rows = old_rows + new_rows
    schedules, exposures, listings = {}, {}, {}
    deps = [OLD_MANIFEST, COLLECTION_BASE/"plan.json", COLLECTION_BASE/"pilot/collection-receipt.json", COLLECTION_BASE/"pilot-review.json", Path(__file__).resolve(), WEIGHTS]
    for seed in SEEDS:
        key = f"visibility-452-{seed}"; seq = _schedule(old_rows, new_rows, seed); schedules[key] = seq; exposures[key] = _exposure(rows, seq)
        listing = OUT / f"{key}.txt"; text = "".join(f"{next(r['image_path'] for r in rows if r['member_id']==m)}\n" for m in sorted(set(seq)))
        if listing.exists() and listing.read_text() != text: raise ValueError("Listing drift")
        if not listing.exists(): listing.write_text(text)
        listings[key] = str(listing); deps.append(listing)
    protocol = dict(schema_version=1, status="frozen_source_isolated_visibility_training_preflight_pending", members=rows, pool_rows=rows,
        schedules=schedules, exposures=exposures, listings=listings, names=NAMES, seeds=list(SEEDS),
        initialization={"path": str(WEIGHTS), "sha256": file_sha256(WEIGHTS), "identity": "independent_v2.11"},
        training_config={"steps": STEPS, "batch": BATCH, "imgsz": 640, "device": "cpu", "optimizer": "AdamW", "lr0": 0.0005, "warmup_epochs": 0, "enhancement": "all_closed"},
        evaluation={"device":"cpu", "imgsz":640, "confidence":[0.37,0.001], "iou":0.7, "max_det":300, "match_iou":0.5},
        comparison="Append 12 reviewed source-isolated extreme-layout frames to the 376-member reviewed pool; 60 extra exposures per seed. This is a bounded visibility-transfer diagnostic, not a fixed-budget causal comparison.",
        limitations=["Pilot frames include clear, occluded, boundary and far conditions; visual states remain diagnostic-only.", "Shared primitive assets limit independent-asset claims.", "Extra 60 exposures change budget and are not a pure material or occlusion causal effect."],
        training_started=False, training_admitted=False, promotable=False, unseen_scene_status="sealed_not_evaluated",
        inputs={str(d): file_sha256(d) for d in deps})
    OUT.mkdir(parents=True, exist_ok=True)
    return write_record(path, protocol)


def _loader(p, key, owner):
    from scripts.vision.order_retention_runtime import make_dataset
    from torch.utils.data import DataLoader, Sampler
    dataset = make_dataset(p, key)
    rows = {r["member_id"]: r for r in p["pool_rows"]}; mapping = {path: i for i, path in enumerate(dataset.im_files)}
    expected = set(rows[m]["image_path"] for m in p["schedules"][key])
    if set(mapping) != expected or len(mapping) != 388: raise ValueError("Runtime member set changed")
    class Schedule(Sampler):
        def __len__(self): return 60
        def __iter__(self):
            start = owner.epoch * 60
            return iter(mapping[rows[mid]["image_path"]] for mid in p["schedules"][key][start:start+60])
    return dataset, DataLoader(dataset, batch_size=BATCH, sampler=Schedule(), num_workers=0, collate_fn=dataset.collate_fn)


def preflight():
    import torch
    from ultralytics import YOLO
    from ultralytics.utils.torch_utils import init_seeds
    p = freeze(); root = OUT / "actual-preflight"; root.mkdir(exist_ok=True); outputs = []
    for key in [f"visibility-452-{s}" for s in SEEDS]:
        dest = root / f"{key}.json"
        if dest.exists(): outputs.append(read_record(dest)); continue
        seed = int(key.rsplit("-", 1)[1]); init_seeds(seed, deterministic=True); owner = SimpleNamespace(epoch=0); actual=[]; batches=[]
        def forbidden(*args, **kwargs): raise AssertionError("Training operation forbidden in preflight")
        with ExitStack() as stack:
            stack.enter_context(__import__('unittest').mock.patch.object(torch.optim.Optimizer, '__init__', side_effect=forbidden))
            stack.enter_context(__import__('unittest').mock.patch.object(torch.Tensor, 'backward', side_effect=forbidden))
            dataset, loader = _loader(p, key, owner); lookup = {r["image_path"]: r["member_id"] for r in p["pool_rows"]}
            for epoch in range(STEPS // 10):
                owner.epoch=epoch
                for batch in loader:
                    if tuple(batch["img"].shape)!=(6,3,640,640): raise ValueError("Unexpected tensor shape")
                    mids=[lookup[x] for x in batch["im_file"]]; pos=len(actual)
                    if mids != p["schedules"][key][pos:pos+6]: raise ValueError("Actual order mismatch")
                    actual.extend(mids); batches.append({"step":len(batches)+1,"members":mids,"image_tensor_shape":list(batch["img"].shape),"label_count":int(batch["cls"].shape[0])})
        if actual != p["schedules"][key] or len(batches) != STEPS: raise ValueError("Preflight exposure mismatch")
        outputs.append(write_record(dest, dict(status="actual_complete_loader_verified_no_training", key=key, actual=actual, batch_records=batches,
            checked_draws=len(actual), checked_batches=len(batches), optimizer_created=False, backward_executed=False, validation_run=False,
            inputs={str(x):file_sha256(x) for x in [OUT/"protocol.json", Path(__file__).resolve()]})))
        print("PREFLIGHT_COMPLETE", key, len(actual), flush=True)
    ready = OUT / "entry-ready.json"
    if not ready.exists():
        write_record(ready, dict(status="ready_for_training_not_started", cells=[f"visibility-452-{s}" for s in SEEDS], actual_draws_verified=STEPS*BATCH*len(SEEDS),
            optimizer_created=False, backward_executed=False, training_admitted=False, promotable=False,
            inputs={str(x):file_sha256(x) for x in [OUT/"protocol.json"]+[root/f"visibility-452-{s}.json" for s in SEEDS]}))
    return read_record(ready)


if __name__ == "__main__":
    import argparse
    ap=argparse.ArgumentParser(); ap.add_argument("--preflight",action="store_true"); a=ap.parse_args()
    if a.preflight: print(preflight()["status"])
    else: print(freeze()["status"])

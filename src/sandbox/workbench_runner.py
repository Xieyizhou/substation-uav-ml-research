"""Execute a bounded YOLO development experiment for the local workbench."""

from __future__ import annotations

import json
from collections import Counter
from pathlib import Path
import shutil
import resource
import sys
import time

from src.ml import EQUIPMENT_CLASSES
from src.ml.artifacts import file_sha256, object_sha256, write_json
from src.sandbox.workbench_datasets import resolve_workbench_dataset
from src.sandbox.workbench_models import WorkbenchExperimentRecipe
from src.sandbox.workbench_run_guard import (
    exclusive_workbench_run, verified_completed_receipt,
)
from src.sandbox.workbench_receipt import materialize_workbench_receipt
from src.sandbox.workbench_weights import resolve_initial_weights
from src.vision.evaluation.detection_metrics import (
    select_confidence_threshold, threshold_metrics,
)
from src.vision.evaluation.yolo_evaluation import (
    _standard_metrics, collect_predictions,
)
from src.vision.replay.static_runtime import runtime_environment, timing_summary


def update_workbench_status(run_root, recipe, **updates):
    path = Path(run_root) / "status.json"
    value = json.loads(path.read_text()) if path.is_file() else {}
    value.update({"workbench_status_schema_version": 1,
                  "experiment_id": recipe.experiment_id,
                  "recipe_identity_sha256": recipe.recipe_identity_sha256}, **updates)
    write_json(path, value)
    return value


def _evenly(paths, limit):
    paths = sorted(paths)
    if limit is None or len(paths) <= limit:
        return paths
    return [paths[min(len(paths) - 1, ((2 * index + 1) * len(paths)) // (2 * limit))]
            for index in range(limit)]


def _link(source, destination):
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists():
        if source.samefile(destination):
            return "existing"
        raise FileExistsError(f"workbench view target already exists: {destination}")
    try:
        destination.hardlink_to(source)
        return "hardlink"
    except OSError:
        shutil.copy2(source, destination)
        return "copy"


def _prepare_view(dataset, run_root, parameters):
    reviewed_members = None
    if getattr(dataset, "source_type", None) == "reviewed_feedback":
        from src.sandbox.workbench_membership import verify_reviewed_dataset
        reviewed_members = verify_reviewed_dataset(dataset)
    source, root = Path(dataset.dataset_root), Path(run_root) / "view"
    members, modes, coverage = [], {}, {}
    for split, limit_name in (("train", "train_limit"),
                              ("validation", "validation_limit")):
        candidates = [path for path in (source / "images" / split).iterdir() if path.is_file()]
        limit = parameters[limit_name]
        if split == "train" and reviewed_members is not None:
            required = {source / row["image_path"] for row in reviewed_members
                        if row["annotation_source"] == "explicit_review"}
            if limit is not None and len(required) > limit:
                raise ValueError("Training budget cannot include all reviewed feedback; choose a larger preset")
            images = sorted(required) + _evenly([p for p in candidates if p not in required],
                        None if limit is None else limit-len(required))
        else:
            images = _evenly(candidates, limit)
        if not images:
            raise ValueError(f"workbench dataset {split} split is empty")
        counts = Counter()
        for image in images:
            label = source / "labels" / split / f"{image.stem}.txt"
            for row in label.read_text().splitlines():
                if row.strip():
                    counts[int(row.split()[0])] += 1
        missing = [name for index, name in enumerate(EQUIPMENT_CLASSES) if not counts[index]]
        if missing and split == "validation":
            raise ValueError(f"workbench {split} selection lacks required classes: {', '.join(missing)}")
        coverage[split] = {name: counts[index] for index, name in enumerate(EQUIPMENT_CLASSES)}
        for image in images:
            label = source / "labels" / split / f"{image.stem}.txt"
            if not label.is_file():
                raise FileNotFoundError(label)
            mode = _link(image, root / "images" / split / image.name)
            _link(label, root / "labels" / split / label.name)
            modes[mode] = modes.get(mode, 0) + 1
            members.append({"sample_id": image.stem, "split": split,
                            "image_sha256": file_sha256(image),
                            "label_sha256": file_sha256(label)})
    names = "".join(f"  {index}: {name}\n" for index, name in enumerate(EQUIPMENT_CLASSES))
    (root / "dataset.yaml").write_text(
        f"path: {root.resolve()}\ntrain: images/train\nval: images/validation\n"
        f"names:\n{names}", encoding="utf-8",
    )
    write_json(root / "membership.json", members)
    return root, {"membership_sha256": object_sha256(members),
                  "frame_count": len(members), "link_modes": modes,
                  "class_counts": coverage}


def _device(requested):
    if requested != "auto":
        return requested
    try:
        import torch
        return "mps" if torch.backends.mps.is_available() else "cpu"
    except ImportError:
        return "cpu"


def _train(weights, view, run_root, recipe, resume):
    try:
        from ultralytics import YOLO
    except ImportError as error:
        raise RuntimeError("workbench training requires requirements-ml.txt") from error
    parameters = recipe.parameters
    directory = Path(run_root) / "training"
    checkpoint = directory / "weights/last.pt"
    if resume and not checkpoint.is_file():
        raise FileNotFoundError("workbench resume requires last.pt")
    model = YOLO(str(checkpoint if resume else weights))
    previous_epoch_end, durations = time.monotonic(), []
    started = previous_epoch_end
    efficiency = {"recipe_identity_sha256": recipe.recipe_identity_sha256,
                  "freeze": parameters.get("freeze", 0), "resumed": resume,
                  "max_resumable_checkpoint_bytes": 0}

    def training_started(trainer):
        efficiency.update(
            parameter_count=sum(p.numel() for p in trainer.model.parameters()),
            trainable_parameter_count=sum(p.numel() for p in trainer.model.parameters() if p.requires_grad),
            device=str(trainer.device),
        )

    def epoch_finished(trainer):
        nonlocal previous_epoch_end
        epoch = min(parameters["epochs"], int(trainer.epoch) + 1)
        # Ultralytics calls this hook again after final best-model validation.
        if epoch <= efficiency.get("last_completed_epoch", 0):
            return
        efficiency["last_completed_epoch"] = epoch
        now = time.monotonic()
        durations.append(now - previous_epoch_end)
        previous_epoch_end = now
        if checkpoint.is_file():
            efficiency["max_resumable_checkpoint_bytes"] = max(
                efficiency["max_resumable_checkpoint_bytes"], checkpoint.stat().st_size
            )
        eta = sum(durations[-5:]) / len(durations[-5:]) * (
            parameters["epochs"] - epoch
        )
        update_workbench_status(run_root, recipe, state="running", stage="training",
                progress=epoch / parameters["epochs"], epoch=epoch,
                total_epochs=parameters["epochs"], eta_seconds=max(0, eta),
                checkpoint_available=checkpoint.is_file())

    model.add_callback("on_train_start", training_started)
    model.add_callback("on_fit_epoch_end", epoch_finished)
    if resume:
        model.train(resume=True)
    else:
        model.train(
            data=str(view / "dataset.yaml"), project=str(Path(run_root)),
            name="training", exist_ok=True, imgsz=parameters["imgsz"],
            epochs=parameters["epochs"], patience=parameters["patience"],
            batch=parameters["batch"], workers=parameters["workers"],
            device=_device(parameters["device"]), cache=False, seed=7,
            deterministic=True, amp=False, optimizer="AdamW", lr0=0.001,
            lrf=0.01, weight_decay=0.0005, warmup_epochs=3, cos_lr=True,
            flipud=0.0, perspective=0.0, mixup=0.0, copy_paste=0.0,
            close_mosaic=min(10, parameters["epochs"]), verbose=True,
            cls_remap=False,
            freeze=parameters.get("freeze", 0),
        )
    best = directory / "weights/best.pt"
    if not best.is_file():
        raise FileNotFoundError(best)
    peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    efficiency.update(
        elapsed_seconds=time.monotonic() - started,
        epoch_callback_intervals_seconds=durations,
        process_peak_rss_bytes=int(peak if sys.platform == "darwin" else peak * 1024),
        rss_scope="process lifetime; includes imports; excludes child workers",
        final_best_bytes=best.stat().st_size,
    )
    write_json(Path(run_root) / "training_efficiency.json", efficiency)
    return best, checkpoint


def evaluate_workbench_model(best, view, run_root, recipe):
    parameters, device = recipe.parameters, _device(recipe.parameters["device"])
    standard = _standard_metrics(
        best, view / "dataset.yaml", split="val", device=device,
        imgsz=parameters["imgsz"], output_root=Path(run_root) / "validation_artifacts",
    )
    frames = collect_predictions(
        best, view, "validation", device=device, imgsz=parameters["imgsz"],
        batch=max(1, min(parameters["batch"], 16)),
    )
    confidence = select_confidence_threshold(frames)
    result = {"workbench_validation_schema_version": 1,
              "model_sha256": file_sha256(best), "standard_metrics": standard,
              "confidence_evaluation": confidence,
              "frame_count": len(frames)}
    write_json(Path(run_root) / "validation.json", result)
    return result, frames


def _export_and_gate(best, view, run_root, recipe, _pt_frames):
    from ultralytics import YOLO

    size = recipe.parameters["imgsz"]
    exported = Path(YOLO(str(best)).export(
        format="onnx", imgsz=size, batch=1, dynamic=False, simplify=True,
        opset=19, nms=False, device="cpu",
    ))
    onnx = Path(run_root) / "model" / f"model_{size}.onnx"
    onnx.parent.mkdir(parents=True, exist_ok=True)
    # The export is run-local; move it instead of storing a second full model.
    if exported.resolve().parent == Path(best).resolve().parent:
        exported.replace(onnx)
    else:
        shutil.copy2(exported, onnx)
    from src.sandbox.workbench_equivalence import evaluate_workbench_equivalence
    gate = evaluate_workbench_equivalence(
        best, onnx, view, imgsz=size, device=_device(recipe.parameters["device"]),
    )
    write_json(Path(run_root) / "onnx_equivalence.json", gate)
    if not gate["passed"]:
        raise ValueError("workbench PT/ONNX equivalence failed")
    return onnx, None, gate


def replay_workbench_model(onnx, view, run_root, recipe, threshold):
    timings = []
    frames = collect_predictions(
        onnx, view, "validation", device="cpu",
        imgsz=recipe.parameters["imgsz"], batch=1, timings=timings,
    )
    environment, environment_id = runtime_environment()
    value = {"workbench_replay_schema_version": 1,
             "model_sha256": file_sha256(onnx), "frame_count": len(frames),
             "successful_frame_count": len(timings), "failed_frame_count": 0,
             "threshold": threshold, "metrics": threshold_metrics(frames, threshold),
             "prediction_passes": 1,
             "timing_scope": "stream_next_including_decode_preprocess_inference_nms_first_frame_startup",
             "timing": timing_summary(timings), "runtime": environment,
             "runtime_identity_sha256": environment_id}
    value["replay_identity_sha256"] = object_sha256(value)
    write_json(Path(run_root) / "replay.json", value)
    return value


def run_workbench_experiment(
    project_root, imported_root, runs_root, recipe_path, *, resume=False,
):
    project_root, recipe_path = Path(project_root), Path(recipe_path)
    recipe = WorkbenchExperimentRecipe.from_record(json.loads(recipe_path.read_text()))
    run_root = Path(runs_root) / recipe.experiment_id
    completed = verified_completed_receipt(run_root, recipe)
    if completed is not None:
        checkpoint = run_root / "training/weights/last.pt"
        update_workbench_status(
            run_root, recipe, state="complete", stage="complete", progress=1.0,
            eta_seconds=0, error=None, failure_code=None,
            checkpoint_available=checkpoint.is_file(),
        )
        return completed
    with exclusive_workbench_run(run_root):
        return _run_workbench_experiment(
            project_root, imported_root, run_root, recipe, resume,
        )


def _run_workbench_experiment(project_root, imported_root, run_root, recipe, resume):
    dataset = resolve_workbench_dataset(project_root, imported_root, recipe.dataset_id)
    if dataset.dataset_identity_sha256 != recipe.dataset_identity_sha256:
        raise ValueError("workbench dataset identity changed")
    try:
        weights = resolve_initial_weights(project_root, recipe)
        update_workbench_status(run_root, recipe, state="running", stage="view_preparation",
                progress=0.0, error=None, failure_code=None)
        view, view_summary = _prepare_view(dataset, run_root, recipe.parameters)
        write_json(run_root / "view.json", view_summary)
        update_workbench_status(run_root, recipe, state="running", stage="training", progress=0.0)
        best, last = _train(weights, view, run_root, recipe, resume)
        update_workbench_status(run_root, recipe, state="running", stage="validation", progress=0.0,
                checkpoint_available=last.is_file())
        validation, frames = evaluate_workbench_model(best, view, run_root, recipe)
        update_workbench_status(run_root, recipe, state="running", stage="export", progress=0.0)
        onnx, _, gate = _export_and_gate(
            best, view, run_root, recipe, frames
        )
        update_workbench_status(run_root, recipe, state="running", stage="replay", progress=0.0)
        threshold = validation["confidence_evaluation"]["selected"]["threshold"]
        replay = replay_workbench_model(onnx, view, run_root, recipe, threshold)
        from src.sandbox.workbench_comparison import compare_with_baseline
        comparison = compare_with_baseline(
            project_root, view, run_root, recipe, replay
        )
        receipt = materialize_workbench_receipt(
            project_root, run_root, recipe, best, onnx, gate, replay, comparison,
        )
        update_workbench_status(run_root, recipe, state="complete", stage="complete", progress=1.0,
                eta_seconds=0, checkpoint_available=last.is_file())
        return receipt
    except Exception as error:
        update_workbench_status(run_root, recipe, state="failed", stage="failed", progress=0.0,
                error=f"{type(error).__name__}: {error}",
                failure_code=_failure_code(error))
        raise


def _failure_code(error):
    if isinstance(error, FileNotFoundError):
        return "artifact_missing"
    if isinstance(error, (ImportError, RuntimeError)) and "requires" in str(error):
        return "dependency_unavailable"
    if "memory" in str(error).lower():
        return "resource_exhausted"
    return "experiment_failed"

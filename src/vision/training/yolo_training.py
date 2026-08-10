"""Ultralytics YOLO training boundary with frozen configuration."""

from __future__ import annotations

import importlib.metadata
import json
from pathlib import Path
import platform
import shutil
import sys

from src.ml.artifacts import file_sha256, git_commit, object_sha256, write_json
from src.vision.contracts.training_identity import TrainingViewIdentity


CONFIG_FIELDS = {
    "imgsz",
    "epochs",
    "patience",
    "device",
    "batch",
    "workers",
    "cache",
    "seed",
    "deterministic",
    "amp",
    "optimizer",
    "lr0",
    "lrf",
    "weight_decay",
    "warmup_epochs",
    "cos_lr",
    "hsv_h",
    "hsv_s",
    "hsv_v",
    "degrees",
    "translate",
    "scale",
    "shear",
    "perspective",
    "flipud",
    "fliplr",
    "mosaic",
    "mixup",
    "copy_paste",
    "close_mosaic",
    "save_period",
}


def load_training_config(path):
    record = json.loads(Path(path).read_text(encoding="utf-8"))
    if record.get("training_config_schema_version") != 1:
        raise ValueError("unsupported visual training config schema")
    missing = CONFIG_FIELDS - set(record)
    if missing:
        raise ValueError(f"training config missing fields: {sorted(missing)}")
    if record.get("pretrained_weights") != "yolo11n.pt":
        raise ValueError("baseline must use yolo11n.pt")
    expected = record.get("pretrained_weights_sha256")
    if not isinstance(expected, str) or len(expected) != 64:
        raise ValueError("baseline must pin the pretrained weights SHA256")
    if record.get("oom_fallback_batch") != 4 or record["batch"] != 8:
        raise ValueError("baseline batch policy must be 8 with OOM fallback 4")
    if record["imgsz"] != 640:
        raise ValueError("baseline training input must be 640")
    return record


def environment_record():
    packages = {}
    for name in (
        "torch",
        "ultralytics",
        "onnx",
        "onnxruntime",
        "onnxslim",
        "numpy",
        "Pillow",
    ):
        try:
            packages[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            packages[name] = None
    mps = None
    try:
        import torch

        mps = {
            "built": bool(torch.backends.mps.is_built()),
            "available": bool(torch.backends.mps.is_available()),
        }
    except ImportError:
        pass
    return {
        "python": sys.version,
        "platform": platform.platform(),
        "machine": platform.machine(),
        "packages": packages,
        "mps": mps,
    }


def _require_clean_commit(project_root):
    commit = git_commit(project_root)
    if commit == "unknown" or commit.endswith("-dirty"):
        raise ValueError("visual training requires a clean tracked worktree")
    return commit


def train_yolo(
    config_path,
    dataset_root,
    output_root,
    *,
    resume=False,
    smoke=False,
    project_root=None,
):
    config = load_training_config(config_path)
    dataset_root = Path(dataset_root)
    output_root = Path(output_root).resolve()
    identity = TrainingViewIdentity.from_record(
        json.loads(
            (dataset_root / "identity/training_view_identity.json").read_text()
        )
    )
    pretrained_path = Path(config["pretrained_weights"])
    if project_root is not None and not pretrained_path.is_absolute():
        pretrained_path = Path(project_root) / pretrained_path
    if not pretrained_path.is_file():
        raise FileNotFoundError(pretrained_path)
    if file_sha256(pretrained_path) != config["pretrained_weights_sha256"]:
        raise ValueError("pretrained YOLO11n weights SHA256 mismatch")
    commit = _require_clean_commit(project_root)
    try:
        from ultralytics import YOLO
    except ImportError as error:
        raise RuntimeError("visual training requires requirements-ml.txt") from error
    resolved = {key: config[key] for key in CONFIG_FIELDS}
    resolved.update(
        {
            "data": str((dataset_root / "dataset.yaml").resolve()),
            "project": str(output_root.parent),
            "name": output_root.name,
            "exist_ok": True,
            "plots": True,
            "verbose": True,
        }
    )
    if smoke:
        resolved.update(
            {
                "epochs": 1,
                "patience": 1,
                "batch": 4,
                "workers": 2,
                "data": str(dataset_root / "dataset-smoke.yaml"),
                "name": f"{output_root.name}-smoke",
            }
        )
    run_name = resolved["name"]
    run_directory = output_root.parent / run_name
    last = run_directory / "weights/last.pt"
    if resume:
        if not last.is_file():
            raise FileNotFoundError(last)
        model = YOLO(str(last))
        result = model.train(resume=True)
    else:
        if last.exists():
            raise ValueError(
                "training output already contains a checkpoint; use --resume"
            )
        model = YOLO(config["pretrained_weights"])
        try:
            result = model.train(**resolved)
        except RuntimeError as error:
            message = str(error).lower()
            if smoke or not any(
                marker in message for marker in ("out of memory", "mps backend")
            ):
                raise
            archive = output_root.parent / f"{run_name}-failed-batch8"
            if archive.exists():
                raise ValueError(f"OOM archive already exists: {archive}") from error
            if run_directory.exists():
                shutil.move(run_directory, archive)
            try:
                import torch

                torch.mps.empty_cache()
            except (AttributeError, ImportError):
                pass
            resolved["batch"] = config["oom_fallback_batch"]
            model = YOLO(config["pretrained_weights"])
            result = model.train(**resolved)
    trainer = model.trainer
    best = Path(trainer.best)
    last = Path(trainer.last)
    run_root = best.parents[1]
    provenance = {
        "training_run_schema_version": 1,
        "run_id": config["run_id"] + ("-smoke" if smoke else ""),
        "training_view_identity_sha256": identity.training_view_identity_sha256,
        "training_code_commit_sha": commit,
        "pretrained_weights": config["pretrained_weights"],
        "pretrained_weights_sha256": (
            file_sha256(pretrained_path)
        ),
        "resolved_config": resolved,
        "resolved_config_identity": object_sha256(
            {
                "training_arguments": resolved,
                "pretrained_weights": config["pretrained_weights"],
                "pretrained_weights_sha256": config[
                    "pretrained_weights_sha256"
                ],
            }
        ),
        "actual_device": str(trainer.device),
        "environment": environment_record(),
        "best_weights": str(best),
        "best_weights_sha256": file_sha256(best),
        "last_weights": str(last),
        "last_weights_sha256": file_sha256(last),
    }
    write_json(run_root / "training_provenance.json", provenance)
    smoke_gate = None
    if smoke:
        reloaded = YOLO(str(last))
        first_image = next(
            (dataset_root / "images/smoke_validation").glob("*.png")
        )
        predictions = reloaded.predict(
            source=str(first_image),
            imgsz=640,
            conf=0.25,
            device=config["device"],
            verbose=False,
        )
        exported = reloaded.export(
            format="onnx",
            imgsz=640,
            batch=1,
            dynamic=False,
            simplify=True,
            opset=19,
            nms=False,
            device="cpu",
        )
        smoke_gate = {
            "checkpoint_reloaded": True,
            "prediction_result_count": len(predictions),
            "onnx_path": str(exported),
            "onnx_sha256": file_sha256(Path(exported)),
        }
        write_json(run_root / "smoke_gate.json", smoke_gate)
    return {
        "run_root": str(run_root),
        "best_weights": str(best),
        "last_weights": str(last),
        "training_provenance": str(run_root / "training_provenance.json"),
        "metrics": getattr(result, "results_dict", {}),
        "smoke_gate": smoke_gate,
    }

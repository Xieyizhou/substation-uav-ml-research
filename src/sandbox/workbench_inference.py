"""Identity-bound local image inference for verified workbench models."""

from __future__ import annotations

from datetime import datetime, timezone
from functools import lru_cache
import json
from pathlib import Path
from time import perf_counter

from src.ml.artifacts import file_sha256, object_sha256, write_json
from src.sandbox.workbench_models import WorkbenchExperimentRecipe
from src.sandbox.workbench_recipe import IDENTIFIER
from src.sandbox.workbench_run_guard import verified_completed_receipt


CLASS_NAMES = ("transformer", "switchgear", "capacitor_bank", "reactor")
MAX_INPUT_BYTES = 50 * 1024 * 1024
MAX_PIXELS = 50_000_000


def verified_model(run_root):
    root = Path(run_root)
    recipe = WorkbenchExperimentRecipe.from_record(
        json.loads((root / "recipe.json").read_text(encoding="utf-8"))
    )
    receipt = verified_completed_receipt(root, recipe)
    if receipt is None:
        raise ValueError("workbench model has no verified completion receipt")
    validation = json.loads((root / "validation.json").read_text(encoding="utf-8"))
    threshold = validation["confidence_evaluation"]["selected"]["threshold"]
    size = recipe.parameters["imgsz"]
    return {
        "experiment_id": recipe.experiment_id,
        "receipt_identity_sha256": receipt["receipt_identity_sha256"],
        "onnx_model_sha256": receipt["onnx_model_sha256"],
        "onnx_path": root / f"model/model_{size}.onnx",
        "imgsz": size,
        "threshold": threshold,
    }


def list_verified_models(runs_root):
    result = []
    for root in sorted(Path(runs_root).iterdir()) if Path(runs_root).is_dir() else ():
        if not (root / "recipe.json").is_file():
            continue
        try:
            paths = tuple(sorted(
                path for pattern in (
                    "recipe.json", "receipt.json", "validation.json", "replay.json",
                    "onnx_equivalence.json", "training/weights/best.pt", "model/*.onnx",
                ) for path in root.glob(pattern)
            ))
            signature = tuple(
                (path.relative_to(root).as_posix(), path.stat().st_size, path.stat().st_mtime_ns)
                for path in paths
            )
            model = _cached_verified_model(str(root.resolve()), signature)
        except (OSError, KeyError, TypeError, ValueError, json.JSONDecodeError):
            continue
        result.append({key: value for key, value in model.items() if key != "onnx_path"})
    return result


@lru_cache(maxsize=64)
def _cached_verified_model(run_root, signature):
    del signature
    return verified_model(Path(run_root))


def _load_image(source):
    from PIL import Image, UnidentifiedImageError

    path = Path(source)
    if path.suffix.lower() not in {".png", ".jpg", ".jpeg"}:
        raise ValueError("workbench inference accepts PNG or JPEG")
    if not path.is_file() or path.stat().st_size > MAX_INPUT_BYTES:
        raise ValueError("workbench inference image is missing or exceeds 50 MiB")
    try:
        image = Image.open(path)
        image.load()
    except (OSError, UnidentifiedImageError) as error:
        raise ValueError("workbench inference image cannot be decoded") from error
    if image.width * image.height > MAX_PIXELS:
        raise ValueError("workbench inference image exceeds 50 megapixels")
    return image.convert("RGB")


def _ultralytics_predict(model, image):
    from ultralytics import YOLO

    started = perf_counter()
    result = YOLO(str(model["onnx_path"]), task="detect").predict(
        source=image, imgsz=model["imgsz"], conf=model["threshold"],
        device="cpu", verbose=False,
    )[0]
    elapsed = (perf_counter() - started) * 1000.0
    detections = []
    for xyxy, confidence, class_id in zip(
        result.boxes.xyxy.cpu().tolist(),
        result.boxes.conf.cpu().tolist(),
        result.boxes.cls.cpu().tolist(),
    ):
        index = int(class_id)
        if index < 0 or index >= len(CLASS_NAMES):
            raise ValueError("model produced an unsupported class identifier")
        detections.append({
            "class_id": index, "class_name": CLASS_NAMES[index],
            "confidence": float(confidence), "xyxy_pixels": [float(v) for v in xyxy],
        })
    return detections, elapsed


def _selected_run(runs_root, experiment_id):
    value = str(experiment_id)
    if not IDENTIFIER.fullmatch(value) or Path(value).name != value:
        raise ValueError("invalid workbench experiment identifier")
    root = (Path(runs_root) / value).resolve()
    if root.parent != Path(runs_root).resolve():
        raise ValueError("workbench experiment is outside the approved runs root")
    return root


def _annotate(image, detections, output):
    from PIL import ImageDraw

    canvas = image.copy()
    draw = ImageDraw.Draw(canvas)
    colors = ("#007a68", "#d88619", "#315d9a", "#9a3f72")
    for item in detections:
        box, color = item["xyxy_pixels"], colors[item["class_id"]]
        draw.rectangle(box, outline=color, width=3)
        label = f'{item["class_name"]} {item["confidence"]:.2f}'
        left, top = box[0], max(0, box[1] - 18)
        draw.rectangle((left, top, left + 8 * len(label) + 8, top + 18), fill=color)
        draw.text((left + 4, top + 2), label, fill="white")
    canvas.save(output, format="PNG", optimize=False)


def run_image_inference(
    runs_root, primary_id, source, output, comparison_id=None, predictor=None,
):
    runs_root, output = Path(runs_root), Path(output)
    if output.exists():
        raise ValueError("workbench inference output already exists")
    image = _load_image(source)
    output.mkdir(parents=True)
    input_path = output / "input.png"
    image.save(input_path, format="PNG", optimize=False)
    predict = predictor or _ultralytics_predict
    models = [("primary", primary_id)]
    if comparison_id:
        if comparison_id == primary_id:
            raise ValueError("comparison model must differ from primary model")
        models.append(("comparison", comparison_id))
    results = {}
    for role, experiment_id in models:
        model = verified_model(_selected_run(runs_root, experiment_id))
        detections, latency_ms = predict(model, image)
        image_name = f"{role}.png"
        _annotate(image, detections, output / image_name)
        results[role] = {
            **{key: value for key, value in model.items() if key != "onnx_path"},
            "detections": detections, "detection_count": len(detections),
            "latency_ms": latency_ms, "annotated_image": image_name,
            "latency_scope": "model_load_and_first_inference",
        }
    record = {
        "workbench_inference_schema_version": 1,
        "inference_id": output.name,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "formal_evidence": False,
        "input_sha256": file_sha256(input_path),
        "input_width": image.width, "input_height": image.height,
        "results": results,
    }
    record["inference_identity_sha256"] = object_sha256(record)
    write_json(output / "result.json", record)
    return record

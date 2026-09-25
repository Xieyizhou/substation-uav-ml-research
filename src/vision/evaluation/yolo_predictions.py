"""Single-pass YOLO prediction collection with optional per-frame timing."""

from pathlib import Path
import time
from src.ml import EQUIPMENT_CLASSES

PREDICTION_CONFIDENCE_FLOOR = 0.05


def _truth(label_path, width=1920, height=1080, input_size=640):
    rows = []
    scale = min(input_size / width, input_size / height)
    for line in Path(label_path).read_text(encoding="utf-8").splitlines():
        class_id, cx, cy, box_width, box_height = map(float, line.split())
        box_width *= width
        box_height *= height
        cx *= width
        cy *= height
        rows.append(
            {
                "class_name": EQUIPMENT_CLASSES[int(class_id)],
                "bbox": [
                    cx - box_width / 2,
                    cy - box_height / 2,
                    cx + box_width / 2,
                    cy + box_height / 2,
                ],
                "small": box_width * box_height * scale * scale < 32 * 32,
            }
        )
    return rows


def _prediction_rows(result):
    if result.boxes is None:
        return []
    rows = []
    for index, class_id in enumerate(result.boxes.cls.tolist()):
        rows.append(
            {
                "class_name": EQUIPMENT_CLASSES[int(class_id)],
                "confidence": float(result.boxes.conf[index]),
                "bbox": [float(value) for value in result.boxes.xyxy[index].tolist()],
            }
        )
    return rows


def collect_predictions(
    model_path,
    dataset_root,
    partition,
    *,
    device,
    imgsz,
    confidence=PREDICTION_CONFIDENCE_FLOOR,
    batch=1,
    timings=None,
):
    if timings is not None and batch != 1:
        raise ValueError("per-frame timings require prediction batch=1")
    try:
        from ultralytics import YOLO
    except ImportError as error:
        raise RuntimeError("visual evaluation requires requirements-ml.txt") from error
    dataset_root = Path(dataset_root)
    image_root = dataset_root / "images" / partition
    label_root = dataset_root / "labels" / partition
    model = YOLO(str(model_path))
    frames = []
    results = model.predict(
        source=str(image_root),
        stream=True,
        imgsz=imgsz,
        conf=confidence,
        batch=batch,
        iou=0.7,
        device=device,
        rect=False,
        verbose=False,
    )
    for result in _timed_results(results, timings):
        stem = Path(result.path).stem
        height, width = (int(value) for value in result.orig_shape)
        frames.append(
            {
                "sample_id": stem,
                "truth": _truth(
                    label_root / f"{stem}.txt", width=width, height=height,
                    input_size=imgsz,
                ),
                "predictions": _prediction_rows(result),
            }
        )
    return frames


def _timed_results(results, timings):
    iterator = iter(results)
    while True:
        started = time.perf_counter()
        try:
            result = next(iterator)
        except StopIteration:
            return
        if timings is not None:
            timings.append((time.perf_counter() - started) * 1_000)
        yield result



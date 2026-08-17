"""Audit, import, and discover immutable YOLO workbench datasets."""

from __future__ import annotations

from collections import Counter
import json
import math
from pathlib import Path
import shutil

from src.ml import EQUIPMENT_CLASSES
from src.ml.artifacts import file_sha256, object_sha256, write_json
from src.sandbox.workbench_models import WorkbenchDataset


IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg"}
BOX_SERIALIZATION_EPSILON = 1e-6


def _yaml(path: Path):
    try:
        import yaml
    except ImportError as error:
        raise RuntimeError("YOLO import requires PyYAML") from error
    value = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError("dataset.yaml must contain an object")
    return value


def _names(value):
    names = value.get("names")
    if isinstance(names, list):
        return {index: str(name) for index, name in enumerate(names)}
    if isinstance(names, dict):
        return {int(index): str(name) for index, name in names.items()}
    raise ValueError("dataset.yaml must define class names")


def _split_root(config, yaml_path, split):
    base = Path(config.get("path", yaml_path.parent))
    if not base.is_absolute():
        base = (yaml_path.parent / base).resolve()
    relative = config.get(split)
    if not isinstance(relative, str):
        raise ValueError(f"YOLO dataset requires a directory for {split}")
    path = Path(relative)
    path = path if path.is_absolute() else base / path
    if not path.is_dir():
        raise FileNotFoundError(path)
    return path.resolve()


def _labels_root(images_root: Path):
    parts = list(images_root.parts)
    matches = [index for index, part in enumerate(parts) if part == "images"]
    if not matches:
        raise ValueError("YOLO split path must be below an images directory")
    parts[matches[-1]] = "labels"
    return Path(*parts)


def _validate_label(path, mapping, source_names):
    objects, rendered = Counter(), []
    text = path.read_text(encoding="utf-8").strip()
    if not text:
        return objects, ""
    for number, line in enumerate(text.splitlines(), 1):
        fields = line.split()
        if len(fields) != 5:
            raise ValueError(f"{path}:{number}: expected five YOLO fields")
        source_class = int(fields[0])
        if source_class not in source_names or source_class not in mapping:
            raise ValueError(f"{path}:{number}: unmapped class {source_class}")
        values = [float(value) for value in fields[1:]]
        x, y, width, height = values
        if (not all(math.isfinite(value) for value in values)
                or width <= 0 or height <= 0
                or any(value < 0 or value > 1 for value in values)):
            raise ValueError(f"{path}:{number}: invalid normalized box")
        x_min, x_max = x - width / 2, x + width / 2
        y_min, y_max = y - height / 2, y + height / 2
        if (x_min < -BOX_SERIALIZATION_EPSILON
                or x_max > 1 + BOX_SERIALIZATION_EPSILON
                or y_min < -BOX_SERIALIZATION_EPSILON
                or y_max > 1 + BOX_SERIALIZATION_EPSILON):
            raise ValueError(f"{path}:{number}: box extends outside the image")
        x_min, x_max = max(0.0, x_min), min(1.0, x_max)
        y_min, y_max = max(0.0, y_min), min(1.0, y_max)
        normalized = (
            (x_min + x_max) / 2,
            (y_min + y_max) / 2,
            x_max - x_min,
            y_max - y_min,
        )
        target = mapping[source_class]
        target_index = EQUIPMENT_CLASSES.index(target)
        objects[target] += 1
        rendered.append(" ".join(
            [str(target_index), *(f"{value:.8f}" for value in normalized)]
        ))
    return objects, "\n".join(rendered) + "\n"


def _copy_or_link(source, destination):
    destination.parent.mkdir(parents=True, exist_ok=True)
    # Imported sources are user-controlled and may be edited after import.
    # A private copy is required because a hard link would mutate with source.
    shutil.copy2(source, destination)
    return "copy"


def _scan_split(split, images_root, mapping, names, destination):
    from PIL import Image

    labels_root = _labels_root(images_root)
    images = sorted(path for path in images_root.rglob("*")
                    if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES)
    if not images:
        raise ValueError(f"YOLO {split} split contains no images")
    records, class_counts, negatives, modes = [], Counter(), 0, Counter()
    for index, image in enumerate(images):
        relative = image.relative_to(images_root)
        label = labels_root / relative.with_suffix(".txt")
        if not label.is_file():
            raise FileNotFoundError(label)
        try:
            with Image.open(image) as opened:
                opened.verify()
        except Exception as error:
            raise ValueError(f"invalid image: {image}") from error
        objects, normalized = _validate_label(label, mapping, names)
        sample_id = f"{split}-{index:06d}"
        target_image = destination / "images" / split / f"{sample_id}{image.suffix.lower()}"
        target_label = destination / "labels" / split / f"{sample_id}.txt"
        modes[_copy_or_link(image, target_image)] += 1
        target_label.parent.mkdir(parents=True, exist_ok=True)
        target_label.write_text(normalized, encoding="utf-8")
        class_counts.update(objects)
        negatives += int(not objects)
        records.append({"sample_id": sample_id, "split": split,
                        "image_sha256": file_sha256(target_image),
                        "label_sha256": file_sha256(target_label)})
    return records, dict(class_counts), negatives, dict(modes)


def import_yolo_dataset(source, output_root, dataset_id, class_mapping=None):
    source, output = Path(source).resolve(), Path(output_root).resolve()
    if not dataset_id or Path(dataset_id).name != dataset_id:
        raise ValueError("invalid workbench dataset identifier")
    yaml_path = source / "dataset.yaml"
    config, names = _yaml(yaml_path), None
    names = _names(config)
    mapping = class_mapping or {
        index: name for index, name in names.items() if name in EQUIPMENT_CLASSES
    }
    mapping = {int(index): str(name) for index, name in mapping.items()}
    if any(name not in EQUIPMENT_CLASSES for name in mapping.values()):
        raise ValueError("class mapping contains a non-canonical class")
    if len(set(mapping.values())) != len(mapping):
        raise ValueError("class mapping must be one-to-one")
    destination = output / dataset_id
    if destination.exists():
        raise ValueError("workbench dataset identifier already exists")
    results, all_records = {}, []
    try:
        for split in ("train", "validation"):
            source_key = "val" if split == "validation" else "train"
            values = _scan_split(
                split, _split_root(config, yaml_path, source_key), mapping,
                names, destination,
            )
            records, classes, negatives, modes = values
            results[split] = {"count": len(records), "class_counts": classes,
                              "no_target_count": negatives, "link_modes": modes}
            all_records.extend(records)
        hashes = [record["image_sha256"] for record in all_records]
        if len(hashes) != len(set(hashes)):
            raise ValueError("duplicate image content across YOLO splits")
        identity = object_sha256({"members": all_records,
                                  "class_names": EQUIPMENT_CLASSES})
        dataset = WorkbenchDataset(
            dataset_id=dataset_id, source_type="imported_yolo",
            dataset_root=str(destination), dataset_identity_sha256=identity,
            split_counts={key: value["count"] for key, value in results.items()},
            class_counts={key: value["class_counts"] for key, value in results.items()},
            no_target_counts={key: value["no_target_count"] for key, value in results.items()},
        )
        write_json(destination / "membership.json", all_records)
        write_json(destination / "dataset.json", dataset.to_record())
        _write_dataset_yaml(destination)
        return {**dataset.to_record(), "import": results}
    except Exception:
        if destination.exists():
            shutil.rmtree(destination)
        raise


def _write_dataset_yaml(root):
    names = "\n".join(f"  {index}: {name}" for index, name in enumerate(EQUIPMENT_CLASSES))
    (root / "dataset.yaml").write_text(
        f"path: {root}\ntrain: images/train\nval: images/validation\nnames:\n{names}\n",
        encoding="utf-8",
    )


def _native_datasets(project_root):
    for identity_path in sorted(Path(project_root).glob(
            "data/research/visual_yolo*/identity/training_view_identity.json")):
        try:
            value = json.loads(identity_path.read_text(encoding="utf-8"))
            root = identity_path.parents[1]
            identity = value["training_view_identity_sha256"]
            yield WorkbenchDataset(
                dataset_id=root.name, source_type="native_training_view",
                dataset_root=str(root), dataset_identity_sha256=identity,
                source_identity_sha256=identity,
                split_counts={"train": int(value["train_frame_count"]),
                              "validation": int(value["validation_frame_count"])},
                class_counts={"train": value.get("train_class_counts", {}),
                              "validation": value.get("validation_class_counts", {})},
                no_target_counts={"train": int(value.get("train_no_target_count", 0)),
                                  "validation": int(value.get("validation_no_target_count", 0))},
            )
        except (KeyError, OSError, TypeError, ValueError):
            continue


def list_workbench_datasets(project_root, imported_root):
    rows = {row.dataset_id: row for row in _native_datasets(project_root)}
    for path in sorted(Path(imported_root).glob("*/dataset.json")):
        try:
            row = WorkbenchDataset.from_record(json.loads(path.read_text(encoding="utf-8")))
            rows[row.dataset_id] = row
        except (KeyError, OSError, TypeError, ValueError):
            continue
    return [rows[key].to_record() for key in sorted(rows)]


def resolve_workbench_dataset(project_root, imported_root, dataset_id):
    matches = [WorkbenchDataset.from_record(row) for row in
               list_workbench_datasets(project_root, imported_root)
               if row["dataset_id"] == dataset_id]
    if len(matches) != 1:
        raise FileNotFoundError(f"unknown workbench dataset: {dataset_id}")
    return matches[0]

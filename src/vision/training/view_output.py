"""Filesystem and identity output for deterministic visual training views."""

from __future__ import annotations

from src.ml import EQUIPMENT_CLASSES
from src.ml.artifacts import file_sha256, write_json
from src.vision.contracts.identity import class_order_identity
from src.vision.contracts.training_identity import TrainingViewIdentity
from src.vision.training.view import evenly_select
from src.vision.training.yolo_dataset import materialize_yolo_partition


def _partitions(collection_root, output_root, train, validation, full_validation):
    rows = {
        "train": train,
        "validation": validation,
        "full_validation": full_validation,
        "smoke_train": evenly_select(train, min(256, len(train))),
        "smoke_validation": evenly_select(validation, min(64, len(validation))),
    }
    return {
        name: materialize_yolo_partition(name, values, collection_root, output_root)
        for name, values in rows.items()
    }


def _write_labels(output_root, results, selection_summary):
    record = {
        name: results[name]["labels_manifest"]
        for name in ("train", "validation", "full_validation")
    }
    if selection_summary is not None:
        record["selection_summary"] = selection_summary
    path = output_root / "identity/labels_manifest.json"
    write_json(path, record)
    return path


def _write_yamls(output_root):
    names = "".join(
        f"  {index}: {name}\n" for index, name in enumerate(EQUIPMENT_CLASSES)
    )
    dataset = output_root / "dataset.yaml"
    dataset.write_text(
        f"path: {output_root.resolve()}\n"
        "train: images/train\nval: images/validation\ntest: images/full_validation\n"
        "names:\n" + names,
        encoding="utf-8",
    )
    smoke = output_root / "dataset-smoke.yaml"
    smoke.write_text(
        f"path: {output_root.resolve()}\n"
        "train: images/smoke_train\nval: images/smoke_validation\n"
        "names:\n" + names,
        encoding="utf-8",
    )
    return dataset, smoke


def _identity(development, validation_source, algorithm, seed, results, labels):
    train, validation = results["train"], results["validation"]
    return TrainingViewIdentity(
        source_development_dataset_identity=development.dataset_identity_sha256,
        source_validation_dataset_identity=(
            None
            if validation_source is None
            else validation_source.dataset_identity_sha256
        ),
        sampling_algorithm=algorithm,
        sampling_seed=seed,
        train_membership_sha256=train["membership_sha256"],
        validation_membership_sha256=validation["membership_sha256"],
        full_validation_membership_sha256=results["full_validation"][
            "membership_sha256"
        ],
        labels_manifest_sha256=file_sha256(labels),
        class_order_identity=class_order_identity(),
        train_frame_count=train["frame_count"],
        validation_frame_count=validation["frame_count"],
        full_validation_frame_count=results["full_validation"]["frame_count"],
        train_class_counts=train["class_counts"],
        validation_class_counts=validation["class_counts"],
        train_no_target_count=train["no_target_count"],
        validation_no_target_count=validation["no_target_count"],
    )


def materialize_view_output(
    collection_root, output_root, development, validation_source,
    train, validation, full_validation, algorithm, seed, selection_summary,
):
    results = _partitions(
        collection_root, output_root, train, validation, full_validation
    )
    labels = _write_labels(output_root, results, selection_summary)
    identity = _identity(
        development, validation_source, algorithm, seed, results, labels
    )
    identity_path = output_root / "identity/training_view_identity.json"
    write_json(identity_path, identity.to_record())
    dataset, smoke = _write_yamls(output_root)
    return {
        "identity": identity.to_record(),
        "identity_path": str(identity_path),
        "dataset_yaml": str(dataset),
        "smoke_dataset_yaml": str(smoke),
        "link_modes": {
            name: result["link_modes"] for name, result in results.items()
        },
    }

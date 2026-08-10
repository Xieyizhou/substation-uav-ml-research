"""Protocol-specific dataset groups and aggregate collection gates."""

from collections import Counter

from src.ml import EQUIPMENT_CLASSES
from src.vision.contracts.protocol import V2_PROTOCOL_ID


V1_DATASET_GROUPS = {
    "development": ("train", "validation"),
    "held_out_test": ("test",),
}
V2_DATASET_GROUPS = {
    "development": ("development",),
    "validation": ("validation",),
    "held_out_test": ("blind",),
}
SIZE_BINS = ("small", "medium", "large")


def dataset_groups(protocol):
    return (
        V2_DATASET_GROUPS
        if protocol["protocol_id"] == V2_PROTOCOL_ID
        else V1_DATASET_GROUPS
    )


def _size_bin(relative_area):
    if relative_area < 0.02:
        return "small"
    if relative_area < 0.10:
        return "medium"
    return "large"


def _split_summary(recordings, split):
    selected = [item for item in recordings if item["row"]["split"] == split]
    classes = Counter()
    size_bins = {name: Counter() for name in EQUIPMENT_CLASSES}
    labelled = 0
    no_target = 0
    for item in selected:
        for annotation in item["annotations"]:
            status = annotation["annotation_status"]
            labelled += status == "labelled"
            no_target += status == "verified_no_target"
            width = annotation["image_width"]
            height = annotation["image_height"]
            by_class = {
                name: [obj for obj in annotation["objects"] if obj["class_name"] == name]
                for name in EQUIPMENT_CLASSES
            }
            for name, objects in by_class.items():
                if not objects:
                    continue
                classes[name] += 1
                largest = max(
                    (obj["bbox_xyxy"][2] - obj["bbox_xyxy"][0])
                    * (obj["bbox_xyxy"][3] - obj["bbox_xyxy"][1])
                    / (width * height)
                    for obj in objects
                )
                size_bins[name][_size_bin(largest)] += 1
    return {
        "recording_count": len(selected),
        "dataset_frame_count": sum(len(item["annotations"]) for item in selected),
        "labelled_frames": labelled,
        "verified_no_target_frames": no_target,
        "classes": {name: classes[name] for name in EQUIPMENT_CLASSES},
        "class_size_bins": {
            name: {size: size_bins[name][size] for size in SIZE_BINS}
            for name in EQUIPMENT_CLASSES
        },
    }


def aggregate_split_counts(recordings, protocol):
    splits = {
        split
        for group in dataset_groups(protocol).values()
        for split in group
    }
    return {
        split: _split_summary(recordings, split)
        for split in sorted(splits)
    }


def _v1_coverage_failures(split_counts, gate):
    failures = []
    minimum = int(gate["minimum_labelled_frames_per_class_per_split"])
    no_target_minimum = int(gate["minimum_verified_no_target_frames_per_split"])
    for split in ("train", "validation", "test"):
        counts = split_counts.get(split, {})
        for class_name in gate["required_classes_per_split"]:
            observed = int(counts.get("classes", {}).get(class_name, 0))
            if observed < minimum:
                failures.append(
                    f"{split} {class_name} labelled frames {observed} below {minimum}"
                )
        observed = int(counts.get("verified_no_target_frames", 0))
        if observed < no_target_minimum:
            failures.append(
                f"{split} verified no-target frames {observed} below {no_target_minimum}"
            )
    return failures


def _v2_coverage_failures(split_counts, gate):
    failures = []
    for split in ("development", "validation", "blind"):
        counts = split_counts.get(split, {})
        class_minimum = int(gate["minimum_labelled_frames_per_class"][split])
        size_minimum = int(gate["minimum_frames_per_class_size_bin"][split])
        for class_name in gate["required_classes_per_split"]:
            observed = int(counts.get("classes", {}).get(class_name, 0))
            if observed < class_minimum:
                failures.append(
                    f"{split} {class_name} labelled frames {observed} below {class_minimum}"
                )
            for size in SIZE_BINS:
                observed = int(
                    counts.get("class_size_bins", {})
                    .get(class_name, {})
                    .get(size, 0)
                )
                if observed < size_minimum:
                    failures.append(
                        f"{split} {class_name} {size} frames {observed} below {size_minimum}"
                    )
        observed = int(counts.get("verified_no_target_frames", 0))
        required = int(gate["minimum_verified_no_target_frames"][split])
        if observed < required:
            failures.append(
                f"{split} verified no-target frames {observed} below {required}"
            )
    return failures


def validate_aggregate_coverage(split_counts, protocol):
    gate = protocol["aggregate_gate"]
    failures = (
        _v2_coverage_failures(split_counts, gate)
        if protocol["protocol_id"] == V2_PROTOCOL_ID
        else _v1_coverage_failures(split_counts, gate)
    )
    if failures:
        raise ValueError("; ".join(failures))

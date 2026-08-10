"""Phase- and object-size-balanced sampling for visual protocol v2."""

from __future__ import annotations

from collections import Counter, defaultdict

from src.ml import EQUIPMENT_CLASSES
from src.vision.training.view import _select_grouped


ALGORITHM = "recording_phase_size_balanced_v2"
TRAIN_SIZE_TARGETS = {"small": 2_000, "medium": 1_500, "large": 1_500}
VALIDATION_SIZE_TARGETS = {"small": 800, "medium": 600, "large": 600}
SELECTION_CLASS_ORDER = tuple(reversed(EQUIPMENT_CLASSES))


def _sequence(row):
    return int(row["annotation"]["sequence_number"])


def _classes(row):
    return {obj["class_name"] for obj in row["annotation"]["objects"]}


def _size_bin(row, class_name):
    annotation = row["annotation"]
    image_area = annotation["image_width"] * annotation["image_height"]
    ratios = []
    for obj in annotation["objects"]:
        if obj["class_name"] != class_name:
            continue
        x1, y1, x2, y2 = obj["bbox_xyxy"]
        ratios.append(max(0.0, x2 - x1) * max(0.0, y2 - y1) / image_area)
    ratio = max(ratios)
    if ratio < 0.02:
        return "small"
    return "medium" if ratio < 0.10 else "large"


def _decimate(rows, grouping):
    groups = defaultdict(list)
    for row in rows:
        groups[grouping(row)].append(row)
    result = []
    for key in sorted(groups, key=str):
        previous = None
        for row in sorted(groups[key], key=lambda item: (_sequence(item), item["sample_id"])):
            sequence = _sequence(row)
            if previous is None or sequence - previous >= 3:
                result.append(row)
                previous = sequence
    return result


def _class_selection(rows, class_name, size_targets):
    candidates = _decimate(
        [row for row in rows if class_name in _classes(row)],
        lambda row: row["recording_id"],
    )
    selected = {}
    available = Counter(_size_bin(row, class_name) for row in candidates)
    for size_name, target in size_targets.items():
        sized = [row for row in candidates if _size_bin(row, class_name) == size_name]
        for row in _select_grouped(
            sized,
            target,
            lambda item: (item["recording_id"], item["annotation"]["mission_phase"]),
        ):
            selected[row["sample_id"]] = row
    cap = sum(size_targets.values())
    if len(selected) < cap:
        remainder = [row for row in candidates if row["sample_id"] not in selected]
        for row in _select_grouped(
            remainder,
            cap - len(selected),
            lambda item: (item["recording_id"], item["annotation"]["mission_phase"]),
        ):
            selected[row["sample_id"]] = row
    summary = {
        "available": {name: available[name] for name in size_targets},
        "requested": dict(size_targets),
        "selected": dict(Counter(_size_bin(row, class_name) for row in selected.values())),
        "shortfall": max(0, cap - len(selected)),
    }
    return selected.values(), summary


def _add_labelled(row, selected, sequences, counts, class_cap):
    sequence = _sequence(row)
    keys = [(row["recording_id"], name) for name in _classes(row)]
    if any(counts[name] >= class_cap for _, name in keys):
        return
    if any(
        abs(sequence - prior) < 3
        for key in keys
        for prior in sequences[key]
    ):
        return
    selected[row["sample_id"]] = row
    for key in keys:
        sequences[key].append(sequence)
        counts[key[1]] += 1


def _summarize_classes(ordered, classes, size_targets):
    class_cap = sum(size_targets.values())
    for class_name in EQUIPMENT_CLASSES:
        actual = Counter(
            _size_bin(row, class_name)
            for row in ordered
            if class_name in _classes(row)
        )
        classes[class_name]["selected"] = {
            size_name: actual[size_name] for size_name in size_targets
        }
        classes[class_name]["shortfall"] = max(
            0, class_cap - sum(actual.values())
        )
    return classes


def select_v2_partition(rows, *, size_targets, negative_cap):
    selected, classes = {}, {}
    sequences, counts = defaultdict(list), Counter()
    class_cap = sum(size_targets.values())

    for class_name in SELECTION_CLASS_ORDER:
        class_rows, summary = _class_selection(rows, class_name, size_targets)
        classes[class_name] = summary
        for row in class_rows:
            if row["sample_id"] not in selected:
                _add_labelled(row, selected, sequences, counts, class_cap)
    negatives = _decimate(
        [
            row
            for row in rows
            if row["annotation"]["annotation_status"] == "verified_no_target"
        ],
        lambda row: row["recording_id"],
    )
    negative_rows = _select_grouped(
        negatives,
        negative_cap,
        lambda item: (item["recording_id"], item["annotation"]["mission_phase"]),
    )
    for row in negative_rows:
        selected[row["sample_id"]] = row
    ordered = sorted(
        selected.values(),
        key=lambda row: (row["recording_id"], _sequence(row), row["sample_id"]),
    )
    return ordered, {
        "classes": _summarize_classes(ordered, classes, size_targets),
        "negative_available": len(negatives),
        "negative_requested": negative_cap,
        "negative_selected": len(negative_rows),
        "negative_shortfall": max(0, negative_cap - len(negative_rows)),
        "deduplicated_frame_count": len(ordered),
    }

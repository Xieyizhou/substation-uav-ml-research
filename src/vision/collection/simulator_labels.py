"""Stable instance labels for Gazebo bounding-box truth."""

from __future__ import annotations

import hashlib


CLASS_LABELS = {
    "transformer": 1,
    "switchgear": 2,
    "capacitor_bank": 3,
    "reactor": 4,
}
CLASS_BY_LABEL = {value: key for key, value in CLASS_LABELS.items()}
INSTANCE_LABEL_STRIDE = 50


def instance_simulator_label(class_name, object_id):
    class_label = CLASS_LABELS.get(class_name)
    if class_label is None:
        return None
    digest = hashlib.sha256(str(object_id).encode("utf-8")).digest()
    slot = int.from_bytes(digest[:4], "big") % (INSTANCE_LABEL_STRIDE - 1) + 1
    return class_label * INSTANCE_LABEL_STRIDE + slot


def class_for_simulator_label(label):
    value = int(label)
    if value in CLASS_BY_LABEL:
        return CLASS_BY_LABEL[value]
    if value <= INSTANCE_LABEL_STRIDE or value >= 5 * INSTANCE_LABEL_STRIDE:
        return None
    return CLASS_BY_LABEL.get((value - 1) // INSTANCE_LABEL_STRIDE)

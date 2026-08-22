"""Validate detection and segmentation labels in quarantined YOLO exports."""

import math


def inspect_label_entries(bundle, entries, class_names):
    invalid = []
    counts = {
        "detection_annotation_count": 0,
        "segmentation_annotation_count": 0,
        "class_image_count": {name: 0 for name in class_names},
        "class_annotation_count": {name: 0 for name in class_names},
        "annotation_conversion": [],
        "polygon_provenance": [],
        "invalid_label_examples": [],
    }
    for entry in entries:
        if entry.file_size > 10 * 1024 * 1024:
            invalid.append({"path": entry.filename, "reason": "label file exceeds size limit"})
            continue
        try:
            text = bundle.read(entry).decode("utf-8")
        except UnicodeDecodeError:
            invalid.append({"path": entry.filename, "reason": "label is not UTF-8"})
            continue
        file_classes = set()
        for line_number, line in enumerate(text.splitlines(), 1):
            if not line.strip():
                continue
            reason, kind, class_name, provenance = _label_row_error(line, class_names)
            if reason:
                invalid.append({"path": entry.filename, "line": line_number, "reason": reason})
                break
            counts["class_annotation_count"][class_name] += 1
            file_classes.add(class_name)
            counts[f"{kind}_annotation_count"] += 1
            if kind == "segmentation":
                conversion = "polygon_to_xyxy_bbox_v1"
                if conversion not in counts["annotation_conversion"]:
                    counts["annotation_conversion"].append(conversion)
                counts["polygon_provenance"].append({
                    "path": entry.filename, "line": line_number,
                    "class_name": class_name, **provenance,
                })
        for name in file_classes:
            counts["class_image_count"][name] += 1
    counts["invalid_label_file_count"] = len(invalid)
    counts["invalid_label_examples"] = invalid[:20]
    counts["annotation_conversion"].sort()
    counts["polygon_provenance"] = counts["polygon_provenance"][:20]
    return counts


def _label_row_error(line, class_names):
    fields = line.split()
    if len(fields) < 5:
        return "expected at least five YOLO fields", None, None, None
    try:
        class_value = float(fields[0])
    except ValueError:
        return "fields must be numeric", None, None, None
    class_id = int(class_value)
    if class_value != class_id or not 0 <= class_id < len(class_names):
        return "class id is outside data.yaml names", None, None, None
    class_name, fields = class_names[class_id], fields[1:]
    if len(fields) == 4:
        return _validate_detection_row(fields, class_name)
    if len(fields) < 6:
        return "polygon rows require at least three points", None, None, None
    if len(fields) % 2:
        return "polygon rows require an even number of coordinate values", None, None, None
    reason, provenance = _validate_polygon_row(fields)
    return (reason, None, None, None) if reason else (None, "segmentation", class_name, provenance)


def _validate_detection_row(fields, class_name):
    try:
        center_x, center_y, width, height = map(float, fields)
    except ValueError:
        return "fields must be numeric", None, None, None
    values = center_x, center_y, width, height
    if not all(math.isfinite(value) for value in values):
        return "coordinates must be finite", None, None, None
    if not all(0.0 <= value <= 1.0 for value in values):
        return "normalized coordinates must be in [0, 1]", None, None, None
    if width <= 0.0 or height <= 0.0:
        return "box width and height must be positive", None, None, None
    if center_x - width / 2 < 0 or center_x + width / 2 > 1:
        return "box extends beyond the image width", None, None, None
    if center_y - height / 2 < 0 or center_y + height / 2 > 1:
        return "box extends beyond the image height", None, None, None
    return None, "detection", class_name, None


def _validate_polygon_row(fields):
    try:
        coordinates = list(map(float, fields))
    except ValueError:
        return "polygon coordinates must be numeric", None
    if not all(math.isfinite(value) for value in coordinates):
        return "polygon coordinates must be finite", None
    if not all(0.0 <= value <= 1.0 for value in coordinates):
        return "polygon coordinates must be in [0, 1]", None
    xs, ys = coordinates[0::2], coordinates[1::2]
    x_min, x_max, y_min, y_max = min(xs), max(xs), min(ys), max(ys)
    if x_min == x_max or y_min == y_max:
        return "polygon area must be greater than zero", None
    return None, {
        "point_count": len(xs), "bbox_xyxy": [x_min, y_min, x_max, y_max],
        "polygon_points": coordinates,
    }

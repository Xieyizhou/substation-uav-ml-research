"""Bounding-box eligibility policies for hard-example selection."""

from dataclasses import replace


def apply_bbox_policy(candidates, policy):
    """Reject frames whose labelled equipment violates split-specific framing rules."""
    if not policy:
        return list(candidates), []
    accepted, rejected = [], []
    for candidate in candidates:
        rules = policy.get(candidate.split, {})
        filter_objects = rules.get("mode") == "drop_invalid_objects"
        target_labels = {
            int(value)
            for value in policy.get("target_instance_labels_by_seed", {}).get(str(candidate.seed), [])
        }
        required_instances = policy.get(
            "required_instance_area_fraction_by_seed", {}
        ).get(str(candidate.seed), [])
        matched_requirements = set()
        margin = float(rules.get("minimum_border_margin_px", 0.0))
        maximum_width = float(rules.get("maximum_bbox_width_fraction", 1.0)) * 1920.0
        maximum_height = float(rules.get("maximum_bbox_height_fraction", 1.0)) * 1080.0
        area_rules = rules.get("bbox_area_fraction_by_class", {})
        kept_objects = []
        dropped_reasons = []
        for item in candidate.objects:
            instance_label = None
            if target_labels or required_instances:
                marker = "-instance-"
                annotation_id = str(item.get("annotation_id", ""))
                try:
                    instance_label = int(annotation_id.split(marker, 1)[1].split("-", 1)[0])
                except (IndexError, ValueError):
                    if target_labels:
                        dropped_reasons.append("missing_instance_lineage")
                        continue
                if instance_label not in target_labels:
                    if target_labels:
                        dropped_reasons.append("non_target_instance")
                        continue
            x1, y1, x2, y2 = map(float, item["bbox_xyxy"])
            class_area_rules = area_rules.get(item.get("class_name"), {})
            area_fraction = ((x2 - x1) * (y2 - y1)) / (1920.0 * 1080.0)
            minimum_area = float(class_area_rules.get("minimum", 0.0))
            maximum_area = float(class_area_rules.get("maximum", 1.0))
            if x1 <= margin or y1 <= margin or x2 >= 1920.0 - margin or y2 >= 1080.0 - margin:
                dropped_reasons.append("bbox_touches_frame_boundary")
            elif x2 - x1 > maximum_width or y2 - y1 > maximum_height:
                dropped_reasons.append("bbox_exceeds_framing_limit")
            elif area_fraction < minimum_area:
                dropped_reasons.append("bbox_area_below_class_minimum")
            elif area_fraction > maximum_area:
                dropped_reasons.append("bbox_area_above_class_maximum")
            else:
                kept_objects.append(item)
                for index, requirement in enumerate(required_instances):
                    if (
                        item.get("class_name") == requirement.get("class_name")
                        and instance_label == int(requirement["instance_label"])
                        and float(requirement["minimum"]) <= area_fraction
                        <= float(requirement["maximum"])
                    ):
                        matched_requirements.add(index)
        if len(matched_requirements) != len(required_instances):
            rejected.append({
                "frame_id": candidate.frame_id,
                "collection": candidate.collection,
                "reason": "required_instance_area_not_satisfied",
            })
            continue
        if filter_objects and dropped_reasons:
            # The RGB still contains these objects. Dropping only their labels
            # creates false negative supervision, including in targeted views.
            rejected.append({
                "frame_id": candidate.frame_id, "collection": candidate.collection,
                "reason": "incomplete_visible_annotations", "scope": "frame",
                "annotation_reasons": dropped_reasons,
                "source_annotation_count": len(candidate.objects),
                "would_remove_annotation_count": len(candidate.objects) - len(kept_objects),
            })
        elif not dropped_reasons:
            accepted.append(candidate)
        else:
            rejected.append({"frame_id": candidate.frame_id, "collection": candidate.collection, "reason": "no_complete_target_annotation" if filter_objects else dropped_reasons[0]})
    return accepted, rejected



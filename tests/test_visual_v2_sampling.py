import types
import unittest
from unittest.mock import patch

from src.ml import EQUIPMENT_CLASSES
from src.vision.training.v2_sampling import SELECTION_CLASS_ORDER, select_v2_partition
from src.vision.training.view_source import V2_PROTOCOL, load_training_sources


def row(sequence, classes=(), recording="recording-a", phase="approach"):
    objects = []
    for index, class_name in enumerate(classes):
        side = 20 + index
        objects.append(
            {
                "class_name": class_name,
                "bbox_xyxy": [10, 10, 10 + side, 10 + side],
            }
        )
    return {
        "sample_id": f"{recording}:{sequence}",
        "recording_id": recording,
        "annotation": {
            "sequence_number": sequence,
            "mission_phase": phase,
            "annotation_status": "labelled" if objects else "verified_no_target",
            "image_width": 100,
            "image_height": 100,
            "objects": objects,
        },
    }


class V2SamplingTests(unittest.TestCase):
    def test_fixed_selection_order_prioritizes_the_scarcer_classes(self):
        self.assertEqual(
            SELECTION_CLASS_ORDER,
            ("reactor", "capacitor_bank", "switchgear", "transformer"),
        )

    def test_multiclass_selection_keeps_three_frame_spacing(self):
        rows = [
            row(index, (EQUIPMENT_CLASSES[index % 2], EQUIPMENT_CLASSES[1]))
            for index in range(1, 16)
        ]
        selected, summary = select_v2_partition(
            rows,
            size_targets={"small": 3, "medium": 0, "large": 0},
            negative_cap=0,
        )
        for class_name in EQUIPMENT_CLASSES[:2]:
            sequences = [
                item["annotation"]["sequence_number"]
                for item in selected
                if class_name
                in {obj["class_name"] for obj in item["annotation"]["objects"]}
            ]
            self.assertTrue(
                all(right - left >= 3 for left, right in zip(sequences, sequences[1:]))
            )
        self.assertEqual(summary["negative_selected"], 0)
        for class_name in EQUIPMENT_CLASSES:
            self.assertLessEqual(
                summary["classes"][class_name]["selected"].get("small", 0)
                + summary["classes"][class_name]["selected"].get("medium", 0)
                + summary["classes"][class_name]["selected"].get("large", 0),
                3,
            )

    def test_source_loader_reads_development_and_validation_only(self):
        development = types.SimpleNamespace(dataset_version=V2_PROTOCOL)
        validation = types.SimpleNamespace(dataset_version=V2_PROTOCOL)
        dev_rows = [{"split": "development"}]
        val_rows = [{"split": "validation"}]
        with patch(
            "src.vision.training.view_source._load_partition",
            side_effect=[(development, dev_rows), (validation, val_rows)],
        ) as loader:
            result = load_training_sources("unused")
        self.assertEqual(result, (development, validation, dev_rows, val_rows))
        self.assertEqual(
            [call.args[1] for call in loader.call_args_list],
            ["development", "validation"],
        )


if __name__ == "__main__":
    unittest.main()

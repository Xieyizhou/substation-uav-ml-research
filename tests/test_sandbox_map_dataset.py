import hashlib
import json
from pathlib import Path
from types import SimpleNamespace
import tempfile
import unittest
from unittest.mock import patch

from PIL import Image

from src.inspection.config import AccessDenied, InspectionConfig
from src.sandbox.map_recording_dataset import (
    VALID_SYNCHRONIZATION_STATUSES,
    register_sandbox_recording,
)
from src.sandbox.workbench_models import WorkbenchDataset
from src.sensors.types import CameraFrame
from src.vision.contracts.annotations import (
    VisualFrameAnnotation,
    VisualObjectAnnotation,
)


class SandboxMapDatasetTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)
        self.recording = self.root / "recording"
        self.datasets = self.root / "datasets"
        (self.recording / "frames").mkdir(parents=True)

    def tearDown(self):
        self.temporary.cleanup()

    def fixtures(self):
        frames, annotations = [], []
        for index in range(6):
            path = self.recording / f"frames/{index:09d}.png"
            Image.new("RGB", (16, 16), (30 + index, 60, 90)).save(path)
            payload_hash = hashlib.sha256(path.read_bytes()).hexdigest()
            frame = CameraFrame(
                frame_id=f"frame-{index}",
                source_id="sandbox-camera",
                sequence_number=index,
                capture_timestamp=float(index),
                capture_clock_domain="simulator",
                receive_monotonic_timestamp=float(index) + 0.1,
                width=16,
                height=16,
                payload_format="png",
                pixel_format="rgb8",
                payload_relative_path=f"frames/{index:09d}.png",
                payload_sha256=payload_hash,
            )
            objects = ()
            status = "verified_no_target"
            if index % 2:
                status = "labelled"
                objects = (VisualObjectAnnotation(
                    annotation_id=f"object-{index}",
                    class_id=0,
                    class_name="transformer",
                    bbox_xyxy=(2.0, 2.0, 14.0, 14.0),
                    visibility_status="visible",
                    truncation_status="not_truncated",
                    truth_source="gazebo_bounding_box_sensor",
                    validation_status="validated",
                ),)
            frames.append(frame)
            annotations.append(VisualFrameAnnotation(
                recording_id="sandbox-recording",
                frame_id=frame.frame_id,
                source_id=frame.source_id,
                sequence_number=index,
                payload_sha256=payload_hash,
                decoded_content_sha256="d" * 64,
                image_width=16,
                image_height=16,
                scenario_id="sandbox-map",
                map_id="custom-map",
                seed=7,
                mission_phase="close_inspection" if objects else "other",
                frame_order_reference=index,
                annotation_status=status,
                objects=objects,
            ))
        metadata = {
            "recording_id": "sandbox-recording",
            "map_revision_identity_sha256": "m" * 64,
        }
        return metadata, frames, annotations, "r" * 64

    def test_registers_an_immutable_development_dataset(self):
        with patch(
            "src.sandbox.map_recording_dataset._recording_data",
            return_value=self.fixtures(),
        ):
            result = register_sandbox_recording(
                self.recording, self.datasets, "custom-map-dataset"
            )
        self.assertEqual(result["source_type"], "sandbox_custom_map")
        self.assertEqual(result["split_counts"], {"validation": 2, "train": 4})
        dataset = WorkbenchDataset.from_record(json.loads(
            (self.datasets / "custom-map-dataset/dataset.json").read_text()
        ))
        self.assertEqual(dataset.no_target_counts, {"validation": 1, "train": 2})
        self.assertEqual(dataset.class_counts["train"]["transformer"], 2)
        receipt = json.loads(
            (self.datasets / "custom-map-dataset/registration_receipt.json").read_text()
        )
        self.assertEqual(receipt["dataset_role"], "development")
        self.assertTrue((self.datasets / "custom-map-dataset/dataset.yaml").is_file())

    def test_rejects_path_injection_and_duplicate_dataset_ids(self):
        fixture = self.fixtures()
        with patch(
            "src.sandbox.map_recording_dataset._recording_data",
            return_value=fixture,
        ):
            with self.assertRaisesRegex(ValueError, "invalid"):
                register_sandbox_recording(self.recording, self.datasets, "../bad")
            register_sandbox_recording(self.recording, self.datasets, "existing")
            with self.assertRaisesRegex(ValueError, "already exists"):
                register_sandbox_recording(self.recording, self.datasets, "existing")

    def test_map_run_paths_remain_inside_the_managed_root(self):
        config = InspectionConfig.for_profile(self.root, "development")
        self.assertEqual(
            config.sandbox_map_run("run-1"),
            (self.root / "outputs/sandbox/map_runs/run-1").resolve(),
        )
        with self.assertRaises(AccessDenied):
            config.sandbox_map_run("../run-1")

    def test_accepts_the_recorder_synchronization_status_vocabulary(self):
        observed = [
            SimpleNamespace(synchronization_status="exact"),
            SimpleNamespace(synchronization_status="nearest_within_tolerance"),
        ]
        self.assertTrue(all(
            item.synchronization_status in VALID_SYNCHRONIZATION_STATUSES
            for item in observed
        ))


if __name__ == "__main__":
    unittest.main()

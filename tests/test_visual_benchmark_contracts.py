from dataclasses import replace
import json
from pathlib import Path
import tempfile
import unittest

import numpy as np

from src.ml.visual_annotations import (
    VisualFrameAnnotation,
    VisualObjectAnnotation,
)
from src.ml.visual_benchmark import (
    TIMING_STAGES,
    VisualBenchmarkCondition,
    VisualBenchmarkResult,
)
from src.ml.visual_benchmark_matrix import validate_static_benchmark_directory
from src.ml.visual_identity import class_order_identity
from src.sensors.camera_decoded import (
    DecodedImage,
    decoded_content_sha256,
)
from src.sensors.types import CameraFrame
from tests.test_visual_identity import (
    HASH_A,
    HASH_B,
    HASH_C,
    HASH_D,
    dataset_identity,
    model_identity,
    preprocessing_identity,
)


ROOT = Path(__file__).resolve().parents[1]
BENCHMARK = ROOT / "benchmarks/visual_static_v1"


def condition(dataset=None, preprocessing=None, model=None, **overrides):
    dataset = dataset or dataset_identity()
    preprocessing = preprocessing or preprocessing_identity()
    model = model or model_identity(preprocessing)
    values = {
        "condition_id": "visual-static-test",
        "dataset_identity_sha256": dataset.dataset_identity_sha256,
        "decoder_configuration_id": dataset.decoder_configuration_id,
        "preprocessing_configuration_id": (
            preprocessing.preprocessing_configuration_id
        ),
        "model_identity_sha256": model.model_identity_sha256,
        "input_width": model.input_width,
        "input_height": model.input_height,
        "inference_policy": "every_frame",
        "frame_skip_interval": 1,
        "target_inference_rate_hz": None,
        "roi_mode": "disabled",
        "batch_size": 1,
        "warmup_frame_count": 2,
        "measured_frame_count": 10,
        "runtime_backend": model.runtime_backend,
        "device_identity": "cpu-test",
        "precision": model.precision,
        "deadline_definition": "source_frame_period",
        "random_seed": 7,
        "software_commit_sha": "test-commit",
    }
    values.update(overrides)
    return VisualBenchmarkCondition(**values)


def timing_summaries(measured_stage=None):
    values = {stage: None for stage in TIMING_STAGES}
    if measured_stage:
        values[measured_stage] = {
            "count": 1,
            "min_ms": 1.0,
            "max_ms": 1.0,
            "mean_ms": 1.0,
            "p50_ms": 1.0,
            "p95_ms": 1.0,
            "p99_ms": 1.0,
        }
    return values


def result_record(**overrides):
    values = {
        "condition_identity_sha256": HASH_A,
        "dataset_identity_sha256": HASH_B,
        "model_identity_sha256": HASH_C,
        "decoder_configuration_id": HASH_D,
        "preprocessing_configuration_id": "e" * 64,
        "software_commit_sha": "test-commit",
        "runtime_environment_identity": "f" * 64,
        "started": True,
        "completion_status": "started",
        "frame_counts": {
            "total": 3,
            "decoded": 1,
            "inferred": 1,
            "skipped": 0,
            "failed_decode": 2,
            "failed_inference": 0,
            "labelled": 1,
        },
        "timing_summaries": timing_summaries("decode_ms"),
        "scheduling_summaries": {
            "requested_inference_frames": 1,
            "completed_inference_frames": 1,
            "deadline_misses": 0,
            "dropped_or_unavailable_frames": 2,
        },
        "visual_metrics": {
            "precision": None,
            "recall": None,
            "mAP50": None,
            "mAP50_95": None,
            "per_class_recall": None,
            "small_object_recall": None,
        },
        "resource_metrics": {"cpu_utilization": None},
        "metric_availability": {
            "precision": False,
            "recall": False,
            "mAP50": False,
            "mAP50_95": False,
            "per_class_recall": False,
            "small_object_recall": False,
            "cpu_utilization": False,
        },
        "unavailable_metrics": (
            "precision",
            "recall",
            "mAP50",
            "mAP50_95",
            "per_class_recall",
            "small_object_recall",
            "cpu_utilization",
        ),
        "failure_codes": ("DECODE_FAILED",),
        "raw_result_artifact_manifest_sha256": None,
    }
    values.update(overrides)
    return values


class AnnotationContractTests(unittest.TestCase):
    def _linked_values(self):
        array = np.array([[[1, 2, 3], [4, 5, 6]]], dtype=np.uint8)
        payload_hash = "a" * 64
        frame = CameraFrame(
            frame_id="frame-1",
            source_id="camera",
            sequence_number=1,
            capture_timestamp=1.0,
            capture_clock_domain="simulator",
            receive_monotonic_timestamp=2.0,
            width=2,
            height=1,
            payload_format="png",
            pixel_format="rgb8",
            payload_relative_path="frames/000000001.png",
            payload_sha256=payload_hash,
        )
        decoded_hash = decoded_content_sha256(array)
        image = DecodedImage(
            frame_id=frame.frame_id,
            source_id=frame.source_id,
            sequence_number=frame.sequence_number,
            width=2,
            height=1,
            pixel_format="rgb8",
            dtype="uint8",
            layout="HWC",
            source_payload_sha256=payload_hash,
            decoded_content_sha256=decoded_hash,
            decoder_id="test-decoder",
            decoder_version="1",
            decoder_configuration_id="b" * 64,
            array=array,
        )
        annotation = VisualFrameAnnotation(
            dataset_identity_sha256="c" * 64,
            frame_id=frame.frame_id,
            source_id=frame.source_id,
            sequence_number=frame.sequence_number,
            payload_sha256=payload_hash,
            decoded_content_sha256=decoded_hash,
            image_width=2,
            image_height=1,
            scenario_id="training-center-2001",
            map_id="training",
            seed=2001,
            mission_phase="close_inspection",
            frame_order_reference=0,
            annotation_status="labelled",
            objects=(
                VisualObjectAnnotation(
                    annotation_id="object-1",
                    class_id=0,
                    class_name="transformer",
                    bbox_xyxy=(0.0, 0.0, 1.0, 1.0),
                    visibility_status="visible",
                    truncation_status="not_truncated",
                    truth_source="simulator_geometry",
                    validation_status="validated",
                ),
            ),
        )
        return frame, image, annotation

    def test_valid_annotation_linkage_round_trips(self):
        frame, image, annotation = self._linked_values()
        self.assertTrue(annotation.validate_linkage(frame, image))
        self.assertEqual(
            VisualFrameAnnotation.from_record(annotation.to_record()), annotation
        )

    def test_invalid_class_box_truth_source_and_dimensions_fail(self):
        with self.assertRaisesRegex(ValueError, "class_id"):
            VisualObjectAnnotation(
                "bad-class",
                1,
                "transformer",
                (0, 0, 1, 1),
                "visible",
                "not_truncated",
                "manual_annotation",
                "validated",
            )
        with self.assertRaisesRegex(ValueError, "positive visible area"):
            VisualObjectAnnotation(
                "bad-box",
                0,
                "transformer",
                (1, 0, 1, 1),
                "visible",
                "not_truncated",
                "manual_annotation",
                "validated",
            )
        with self.assertRaisesRegex(ValueError, "truth source"):
            VisualObjectAnnotation(
                "bad-source",
                0,
                "transformer",
                (0, 0, 1, 1),
                "visible",
                "not_truncated",
                "pseudo_label",
                "validated",
            )
        frame, image, annotation = self._linked_values()
        with self.assertRaisesRegex(ValueError, "linkage mismatch"):
            replace(annotation, image_width=3).validate_linkage(frame, image)

    def test_payload_and_decoded_hash_mismatches_fail_linkage(self):
        frame, image, annotation = self._linked_values()
        for changed in (
            replace(annotation, payload_sha256="d" * 64),
            replace(annotation, decoded_content_sha256="e" * 64),
        ):
            with self.assertRaisesRegex(ValueError, "linkage mismatch"):
                changed.validate_linkage(frame, image)


class BenchmarkConditionTests(unittest.TestCase):
    def test_condition_identity_is_deterministic_and_references_match(self):
        dataset = dataset_identity()
        preprocessing = preprocessing_identity()
        model = model_identity(preprocessing)
        first = condition(dataset, preprocessing, model)
        second = condition(dataset, preprocessing, model)
        self.assertEqual(
            first.condition_identity_sha256,
            second.condition_identity_sha256,
        )
        self.assertEqual(
            VisualBenchmarkCondition.from_record(first.to_record()), first
        )
        self.assertTrue(first.validate_references(dataset, model, preprocessing))

    def test_frame_skip_batch_and_reference_mismatches_fail(self):
        with self.assertRaisesRegex(ValueError, "every_frame"):
            condition(frame_skip_interval=2)
        with self.assertRaisesRegex(ValueError, "batch_size"):
            condition(batch_size=0)
        current = condition()
        with self.assertRaisesRegex(ValueError, "identity mismatch"):
            current.validate_references(
                dataset_identity(decoder_configuration_id=HASH_B),
                model_identity(),
                preprocessing_identity(),
            )


class BenchmarkResultTests(unittest.TestCase):
    def test_unavailable_metrics_remain_null_and_failures_are_separate(self):
        result = VisualBenchmarkResult(**result_record())
        self.assertIsNone(result.visual_metrics["precision"])
        self.assertIsNone(result.resource_metrics["cpu_utilization"])
        self.assertEqual(result.timing_summaries["decode_ms"]["count"], 1)
        self.assertEqual(result.frame_counts["failed_decode"], 2)
        self.assertEqual(
            VisualBenchmarkResult.from_record(result.to_record()), result
        )

    def test_completed_result_requires_artifact_manifest(self):
        with self.assertRaisesRegex(ValueError, "artifact manifest"):
            VisualBenchmarkResult(
                **result_record(completion_status="completed")
            )
        completed = VisualBenchmarkResult(
            **result_record(
                completion_status="completed",
                raw_result_artifact_manifest_sha256=HASH_A,
            )
        )
        self.assertEqual(completed.completion_status, "completed")

    def test_result_identity_hash_detects_reference_changes(self):
        first = VisualBenchmarkResult(**result_record())
        second = VisualBenchmarkResult(
            **result_record(model_identity_sha256=HASH_D)
        )
        self.assertNotEqual(
            first.result_identity_sha256, second.result_identity_sha256
        )

    def test_result_references_exact_condition_and_artifact_identities(self):
        dataset = dataset_identity()
        preprocessing = preprocessing_identity()
        model = model_identity(preprocessing)
        benchmark_condition = condition(dataset, preprocessing, model)
        result = VisualBenchmarkResult(
            **result_record(
                condition_identity_sha256=(
                    benchmark_condition.condition_identity_sha256
                ),
                dataset_identity_sha256=dataset.dataset_identity_sha256,
                model_identity_sha256=model.model_identity_sha256,
                decoder_configuration_id=dataset.decoder_configuration_id,
                preprocessing_configuration_id=(
                    preprocessing.preprocessing_configuration_id
                ),
            )
        )
        self.assertTrue(
            result.validate_references(
                benchmark_condition, dataset, model, preprocessing
            )
        )
        with self.assertRaisesRegex(ValueError, "result identity mismatch"):
            replace(
                result, model_identity_sha256=HASH_A
            ).validate_references(
                benchmark_condition, dataset, model, preprocessing
            )


class StaticMatrixTests(unittest.TestCase):
    def test_matrix_has_nine_unique_unmaterialized_nonadaptive_templates(self):
        result = validate_static_benchmark_directory(BENCHMARK)
        self.assertEqual(result["template_count"], 9)
        ids = [row["template_id"] for row in result["templates"]]
        self.assertEqual(len(ids), len(set(ids)))
        self.assertTrue(
            all(row["materialization_status"] == "template" for row in result["templates"])
        )
        self.assertNotIn(
            "adaptive",
            {row["inference_policy"] for row in result["templates"]},
        )

    def test_template_with_real_identity_or_duplicate_is_rejected(self):
        benchmark = json.loads((BENCHMARK / "benchmark.json").read_text())
        conditions = json.loads((BENCHMARK / "conditions.json").read_text())
        conditions["conditions"][0]["model_identity"] = HASH_A
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "benchmark.json").write_text(json.dumps(benchmark))
            (root / "conditions.json").write_text(json.dumps(conditions))
            for name in (
                "metrics.json",
                "pilot_protocol.json",
                "class_order.json",
            ):
                (root / name).write_text((BENCHMARK / name).read_text())
            with self.assertRaisesRegex(ValueError, "cannot claim"):
                validate_static_benchmark_directory(root)

    def test_locked_class_order_identity_matches_repository_contract(self):
        record = json.loads((BENCHMARK / "class_order.json").read_text())
        self.assertEqual(record["class_order_identity"], class_order_identity())


if __name__ == "__main__":
    unittest.main()

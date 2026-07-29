from dataclasses import replace
import unittest

from src.ml.visual_identity import (
    DatasetIdentity,
    ModelIdentity,
    PreprocessingIdentity,
    class_order_identity,
)


HASH_A = "a" * 64
HASH_B = "b" * 64
HASH_C = "c" * 64
HASH_D = "d" * 64
HASH_E = "e" * 64
HASH_F = "f" * 64


def dataset_identity(**overrides):
    values = {
        "dataset_name": "substation-visual-pilot",
        "dataset_version": "v1",
        "dataset_role": "pilot",
        "recording_schema_version": 1,
        "annotation_schema_version": 1,
        "decoder_configuration_id": HASH_A,
        "recording_manifest_sha256": HASH_B,
        "annotation_manifest_sha256": HASH_C,
        "scenario_manifest_sha256": HASH_D,
        "split_manifest_sha256": HASH_E,
        "ordered_frame_count": 120,
        "labelled_frame_count": 100,
        "source_recording_ids": ("recording-training-center-2001",),
        "scenario_ids": ("training-center-2001",),
        "map_ids": ("training",),
        "seed_ids": (2001,),
        "class_order_identity": class_order_identity(),
        "creation_commit_sha": None,
        "source_payload_formats": ("png",),
    }
    values.update(overrides)
    return DatasetIdentity(**values)


def preprocessing_identity(size=320, **overrides):
    values = {
        "target_input_width": size,
        "target_input_height": size,
        "resize_policy": "letterbox",
        "letterbox_policy": "preserve_aspect_ratio_center",
        "interpolation_method": "bilinear",
        "padding_value": (114, 114, 114),
        "hwc_to_chw": True,
        "rgb_bgr_policy": "preserve_rgb",
        "uint8_to_float": True,
        "normalization_scale": 1.0 / 255.0,
        "mean": None,
        "standard_deviation": None,
        "batch_dimension_policy": "add_single_batch_dimension",
        "tensor_dtype": "float32",
        "contiguous_memory_policy": "require_contiguous",
        "implementation": "visual-preprocess",
        "implementation_version": "1.0.0",
    }
    values.update(overrides)
    return PreprocessingIdentity(**values)


def model_identity(preprocessing=None, **overrides):
    preprocessing = preprocessing or preprocessing_identity()
    values = {
        "model_family": "yolo",
        "architecture_variant": "unmaterialized-test-fixture",
        "task": "equipment_detection",
        "class_order_identity": class_order_identity(),
        "input_width": preprocessing.target_input_width,
        "input_height": preprocessing.target_input_height,
        "input_pixel_format": "rgb8",
        "preprocessing_configuration_id": (
            preprocessing.preprocessing_configuration_id
        ),
        "runtime_backend": "onnxruntime",
        "precision": "fp32",
        "weights_sha256": HASH_A,
        "model_file_sha256": HASH_B,
        "model_package_schema_version": 1,
        "runtime_version_identity": "onnxruntime-test",
        "training_dataset_identity": None,
        "training_code_commit_sha": None,
        "export_configuration_identity": None,
        "onnx_opset": 18,
    }
    values.update(overrides)
    return ModelIdentity(**values)


class DatasetIdentityTests(unittest.TestCase):
    def test_identity_is_deterministic_and_round_trips(self):
        first = dataset_identity()
        second = dataset_identity()
        self.assertEqual(
            first.dataset_identity_sha256, second.dataset_identity_sha256
        )
        self.assertEqual(
            DatasetIdentity.from_record(first.to_record()), first
        )

    def test_content_affecting_components_change_identity(self):
        baseline = dataset_identity()
        variants = (
            replace(baseline, recording_manifest_sha256=HASH_F),
            replace(baseline, annotation_manifest_sha256=HASH_F),
            replace(baseline, decoder_configuration_id=HASH_F),
            replace(baseline, split_manifest_sha256=HASH_F),
            replace(baseline, scenario_manifest_sha256=HASH_F),
            replace(
                baseline,
                class_order_identity=HASH_F,
            ),
        )
        for variant in variants:
            with self.subTest(variant=variant):
                self.assertNotEqual(
                    baseline.dataset_identity_sha256,
                    variant.dataset_identity_sha256,
                )

    def test_ordered_frame_membership_is_bound_by_recording_manifest_hash(self):
        first_order = dataset_identity(recording_manifest_sha256=HASH_A)
        changed_order = dataset_identity(recording_manifest_sha256=HASH_B)
        self.assertNotEqual(
            first_order.dataset_identity_sha256,
            changed_order.dataset_identity_sha256,
        )

    def test_paths_hashes_and_required_annotation_identity_are_validated(self):
        with self.assertRaisesRegex(ValueError, "absolute path"):
            dataset_identity(dataset_name="/tmp/dataset")
        with self.assertRaisesRegex(ValueError, "SHA256"):
            dataset_identity(recording_manifest_sha256="bad")
        with self.assertRaisesRegex(ValueError, "annotation_manifest"):
            dataset_identity(annotation_manifest_sha256=None)
        with self.assertRaises(TypeError):
            DatasetIdentity.from_record({"dataset_identity_sha256": HASH_A})

    def test_png_is_the_canonical_v1_storage_policy(self):
        self.assertEqual(dataset_identity().canonical_payload_format, "png")
        with self.assertRaisesRegex(ValueError, "canonical payload"):
            dataset_identity(canonical_payload_format="jpeg")


class PreprocessingIdentityTests(unittest.TestCase):
    def test_identity_is_deterministic_and_round_trips(self):
        identity = preprocessing_identity()
        self.assertEqual(
            PreprocessingIdentity.from_record(identity.to_record()), identity
        )
        self.assertEqual(
            identity.preprocessing_configuration_id,
            preprocessing_identity().preprocessing_configuration_id,
        )

    def test_interpolation_normalization_and_size_change_identity(self):
        baseline = preprocessing_identity()
        variants = (
            preprocessing_identity(interpolation_method="nearest"),
            preprocessing_identity(normalization_scale=1.0),
            preprocessing_identity(size=640),
        )
        for variant in variants:
            self.assertNotEqual(
                baseline.preprocessing_configuration_id,
                variant.preprocessing_configuration_id,
            )

    def test_unstable_local_values_are_rejected(self):
        with self.assertRaisesRegex(ValueError, "absolute path"):
            preprocessing_identity(implementation="/tmp/preprocessor.py")


class ModelIdentityTests(unittest.TestCase):
    def test_identity_is_deterministic_and_round_trips(self):
        identity = model_identity()
        self.assertEqual(ModelIdentity.from_record(identity.to_record()), identity)
        self.assertEqual(
            identity.model_identity_sha256,
            model_identity().model_identity_sha256,
        )

    def test_weights_size_preprocessing_and_class_order_change_identity(self):
        baseline_preprocess = preprocessing_identity()
        baseline = model_identity(baseline_preprocess)
        larger_preprocess = preprocessing_identity(640)
        variants = (
            model_identity(baseline_preprocess, weights_sha256=HASH_C),
            model_identity(larger_preprocess),
            model_identity(
                baseline_preprocess,
                preprocessing_configuration_id=HASH_D,
            ),
            model_identity(
                baseline_preprocess,
                class_order_identity=HASH_E,
            ),
        )
        for variant in variants:
            self.assertNotEqual(
                baseline.model_identity_sha256, variant.model_identity_sha256
            )

    def test_unknown_optional_provenance_remains_null(self):
        record = model_identity().to_record()
        self.assertIsNone(record["training_dataset_identity"])
        self.assertIsNone(record["training_code_commit_sha"])
        self.assertIsNone(record["export_configuration_identity"])

    def test_filename_and_absolute_path_are_not_model_identity(self):
        with self.assertRaisesRegex(ValueError, "absolute path"):
            model_identity(model_family="/tmp/model.onnx")


if __name__ == "__main__":
    unittest.main()

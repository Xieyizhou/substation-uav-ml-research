from io import BytesIO
import hashlib
import json
from pathlib import Path
import tempfile
import unittest

import numpy as np
from PIL import Image

from src.sensors.camera_decoded import (
    DecodedImage,
    DecoderConfiguration,
    DecoderConfigurationError,
    DecoderContentHashError,
    DecoderDimensionMismatchError,
    DecoderMalformedPayloadError,
    DecoderPayloadFormatMismatchError,
    DecoderPayloadIntegrityError,
    DecoderPayloadMissingError,
    DecoderRawByteCountError,
    DecoderUnsupportedDtypeError,
    DecoderUnsupportedPixelFormatError,
    DecoderUnsupportedStrideError,
    decoded_content_sha256,
)
from src.sensors.camera_decoder import (
    decode_camera_payload,
    validate_decode_determinism,
)
from src.sensors.camera_recording import (
    CAMERA_RECORDING_SCHEMA_VERSION,
    CameraReplaySource,
    record_camera_frames,
)
from src.sensors.types import CAMERA_RAW_LAYOUT, CameraFrame, VisualTiming
from src.sensors.visual_timing import summarize_visual_timings


def _sha256(payload):
    return hashlib.sha256(payload).hexdigest()


def _encoded_payload(array, payload_format):
    output = BytesIO()
    Image.fromarray(array).save(
        output, format={"png": "PNG", "jpeg": "JPEG"}[payload_format]
    )
    return output.getvalue()


def _frame(
    payload,
    *,
    payload_format="png",
    pixel_format="rgb8",
    width=2,
    height=1,
    sequence=0,
    path=None,
    metadata=None,
):
    extension = {"png": ".png", "jpeg": ".jpg", "raw": ".raw"}[payload_format]
    return CameraFrame(
        frame_id=f"frame-{sequence}",
        source_id="decoder-test-camera",
        sequence_number=sequence,
        capture_timestamp=float(sequence),
        capture_clock_domain="simulator",
        receive_monotonic_timestamp=10.0 + sequence,
        width=width,
        height=height,
        payload_format=payload_format,
        pixel_format=pixel_format,
        payload_relative_path=path or f"source-{sequence}{extension}",
        payload_sha256=_sha256(payload),
        metadata=metadata or {},
    )


def _record(root, payload_frames):
    frames = []
    for payload, frame in payload_frames:
        (root / frame.payload_relative_path).write_bytes(payload)
        frames.append(frame)
    recording = root / "recording"
    record_camera_frames(frames, recording, payload_root=root)
    return recording, list(CameraReplaySource(recording).iter_frames())


def _raw_root(root, frame, payload, *, include_layout=True):
    root.mkdir(parents=True, exist_ok=True)
    (root / frame.payload_relative_path).write_bytes(payload)
    metadata = {
        "recording_schema_version": CAMERA_RECORDING_SCHEMA_VERSION,
        "recording_type": "camera_frame_collection",
    }
    if include_layout:
        metadata["raw_layout_contract"] = CAMERA_RAW_LAYOUT
    (root / "metadata.json").write_text(json.dumps(metadata))
    return root


class CanonicalDecodeTests(unittest.TestCase):
    def test_png_decodes_to_read_only_contiguous_rgb8_hwc_uint8(self):
        pixels = np.array([[[1, 2, 3], [4, 5, 6]]], dtype=np.uint8)
        payload = _encoded_payload(pixels, "png")
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            recording, frames = _record(root, [(payload, _frame(payload))])
            result = decode_camera_payload(frames[0], recording)
        image = result.image
        self.assertEqual(image.pixel_format, "rgb8")
        self.assertEqual(image.dtype, "uint8")
        self.assertEqual(image.layout, "HWC")
        self.assertEqual(image.array.shape, (1, 2, 3))
        self.assertEqual(image.array.dtype, np.uint8)
        self.assertTrue(image.array.flags.c_contiguous)
        self.assertFalse(image.array.flags.writeable)
        np.testing.assert_array_equal(image.array, pixels)
        self.assertNotIn("array", image.to_metadata_record())

    def test_grayscale_is_repeated_and_alpha_is_dropped_without_compositing(self):
        gray = np.array([[7, 9]], dtype=np.uint8)
        rgba = np.array([[[10, 20, 30, 0], [40, 50, 60, 255]]], dtype=np.uint8)
        gray_payload = _encoded_payload(gray, "png")
        rgba_payload = _encoded_payload(rgba, "png")
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            recording, frames = _record(
                root,
                [
                    (
                        gray_payload,
                        _frame(
                            gray_payload,
                            pixel_format="mono8",
                            sequence=0,
                        ),
                    ),
                    (
                        rgba_payload,
                        _frame(
                            rgba_payload,
                            pixel_format="rgba8",
                            sequence=1,
                        ),
                    ),
                ],
            )
            gray_result = decode_camera_payload(frames[0], recording)
            rgba_result = decode_camera_payload(frames[1], recording)
        np.testing.assert_array_equal(
            gray_result.image.array,
            np.array([[[7, 7, 7], [9, 9, 9]]], dtype=np.uint8),
        )
        np.testing.assert_array_equal(rgba_result.image.array, rgba[:, :, :3])

    def test_bgr_raw_converts_explicitly_to_rgb(self):
        bgr = np.array([[[10, 20, 30], [1, 2, 3]]], dtype=np.uint8)
        payload = bgr.tobytes()
        frame = _frame(payload, payload_format="raw", pixel_format="bgr8")
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            recording, frames = _record(root, [(payload, frame)])
            result = decode_camera_payload(frames[0], recording)
        np.testing.assert_array_equal(
            result.image.array,
            np.array([[[30, 20, 10], [3, 2, 1]]], dtype=np.uint8),
        )

    def test_jpeg_is_supported_but_still_produces_canonical_rgb(self):
        pixels = np.array([[[20, 40, 60], [80, 100, 120]]], dtype=np.uint8)
        payload = _encoded_payload(pixels, "jpeg")
        frame = _frame(payload, payload_format="jpeg")
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            recording, frames = _record(root, [(payload, frame)])
            result = decode_camera_payload(frames[0], recording)
        self.assertEqual(result.image.array.shape, (1, 2, 3))
        self.assertEqual(result.image.pixel_format, "rgb8")


class DecoderIntegrityTests(unittest.TestCase):
    def test_missing_payload_has_typed_error_with_portable_frame_identity(self):
        payload = b"missing"
        frame = _frame(payload)
        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaises(DecoderPayloadMissingError) as raised:
                decode_camera_payload(frame, directory)
        message = str(raised.exception)
        self.assertIn("frame_id='frame-0'", message)
        self.assertIn("sequence=0", message)
        self.assertNotIn(directory, message)

    def test_source_hash_is_verified_before_decode(self):
        payload = _encoded_payload(
            np.array([[[1, 2, 3], [4, 5, 6]]], dtype=np.uint8), "png"
        )
        frame = _frame(payload)
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / frame.payload_relative_path).write_bytes(payload + b"corrupt")
            with self.assertRaises(DecoderPayloadIntegrityError):
                decode_camera_payload(frame, root)

    def test_declared_format_extension_and_content_must_agree(self):
        payload = _encoded_payload(
            np.array([[[1, 2, 3], [4, 5, 6]]], dtype=np.uint8), "png"
        )
        extension_mismatch = _frame(payload, path="frame.jpg")
        content_mismatch = _frame(
            payload,
            payload_format="jpeg",
            path="frame.jpg",
        )
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "frame.jpg").write_bytes(payload)
            with self.assertRaises(DecoderPayloadFormatMismatchError):
                decode_camera_payload(extension_mismatch, root)
            with self.assertRaises(DecoderPayloadFormatMismatchError):
                decode_camera_payload(content_mismatch, root)

    def test_malformed_payload_dimension_and_mode_mismatches_fail(self):
        malformed = b"not-a-png"
        malformed_frame = _frame(malformed)
        rgb_payload = _encoded_payload(
            np.array([[[1, 2, 3], [4, 5, 6]]], dtype=np.uint8), "png"
        )
        wrong_dimensions = _frame(rgb_payload, width=3)
        wrong_mode = _frame(rgb_payload, pixel_format="mono8")
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for frame, payload, error_type in (
                (malformed_frame, malformed, DecoderMalformedPayloadError),
                (wrong_dimensions, rgb_payload, DecoderDimensionMismatchError),
                (wrong_mode, rgb_payload, DecoderUnsupportedPixelFormatError),
            ):
                (root / frame.payload_relative_path).write_bytes(payload)
                with self.assertRaises(error_type):
                    decode_camera_payload(frame, root)

    def test_decoded_hash_covers_content_shape_and_metadata(self):
        first = np.array([[[1, 2, 3], [4, 5, 6]]], dtype=np.uint8)
        changed = first.copy()
        changed[0, 0, 0] = 99
        reshaped = first.reshape(2, 1, 3)
        first_hash = decoded_content_sha256(first)
        self.assertEqual(first_hash, decoded_content_sha256(first.copy()))
        self.assertNotEqual(first_hash, decoded_content_sha256(changed))
        self.assertNotEqual(first_hash, decoded_content_sha256(reshaped))
        self.assertNotEqual(
            first_hash,
            decoded_content_sha256(first, pixel_format="bgr8"),
        )

    def test_decoded_contract_rejects_wrong_content_hash(self):
        array = np.zeros((1, 1, 3), dtype=np.uint8)
        with self.assertRaises(DecoderContentHashError):
            DecodedImage(
                frame_id="frame",
                source_id="camera",
                sequence_number=1,
                width=1,
                height=1,
                pixel_format="rgb8",
                dtype="uint8",
                layout="HWC",
                source_payload_sha256="a" * 64,
                decoded_content_sha256="b" * 64,
                decoder_id="decoder",
                decoder_version="1",
                decoder_configuration_id="c" * 64,
                array=array,
            )


class RawValidationTests(unittest.TestCase):
    def test_invalid_raw_count_missing_layout_and_stride_fail(self):
        payload = b"\x01\x02\x03\x04\x05"
        frame = _frame(payload, payload_format="raw", pixel_format="rgb8")
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            _raw_root(root, frame, payload)
            with self.assertRaises(DecoderRawByteCountError):
                decode_camera_payload(frame, root)
        valid = bytes(range(6))
        frame = _frame(valid, payload_format="raw", pixel_format="rgb8")
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            _raw_root(root, frame, valid, include_layout=False)
            with self.assertRaises(DecoderUnsupportedStrideError):
                decode_camera_payload(frame, root)
        stride_frame = _frame(
            valid,
            payload_format="raw",
            pixel_format="rgb8",
            metadata={"row_stride_bytes": 99},
        )
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            _raw_root(root, stride_frame, valid)
            with self.assertRaises(DecoderUnsupportedStrideError):
                decode_camera_payload(stride_frame, root)

    def test_unsupported_raw_dtype_fails(self):
        payload = bytes(range(6))
        frame = _frame(
            payload,
            payload_format="raw",
            metadata={"dtype": "float32"},
        )
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            _raw_root(root, frame, payload)
            with self.assertRaises(DecoderUnsupportedDtypeError):
                decode_camera_payload(frame, root)

    def test_rgb_and_bgr_raw_are_not_silently_interchanged(self):
        payload = bytes([1, 2, 3, 4, 5, 6])
        rgb_frame = _frame(payload, payload_format="raw", pixel_format="rgb8")
        bgr_frame = _frame(payload, payload_format="raw", pixel_format="bgr8")
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            _raw_root(root, rgb_frame, payload)
            rgb = decode_camera_payload(rgb_frame, root).image.array
            bgr = decode_camera_payload(bgr_frame, root).image.array
        self.assertFalse(np.array_equal(rgb, bgr))
        np.testing.assert_array_equal(rgb[0, 0], [1, 2, 3])
        np.testing.assert_array_equal(bgr[0, 0], [3, 2, 1])


class DecoderDeterminismAndTimingTests(unittest.TestCase):
    def test_configuration_identity_is_stable_and_rejects_unknown_schema(self):
        first = DecoderConfiguration()
        second = DecoderConfiguration()
        self.assertEqual(first.configuration_id, second.configuration_id)
        self.assertEqual(len(first.configuration_id), 64)
        with self.assertRaises(DecoderConfigurationError):
            DecoderConfiguration(schema_version=999)
        with self.assertRaises(DecoderConfigurationError):
            DecoderConfiguration(options={"cpu_only": "false"})

    def test_png_and_raw_repeated_decodes_are_deterministic_and_ordered(self):
        png_pixels = np.array([[[1, 2, 3], [4, 5, 6]]], dtype=np.uint8)
        png_payload = _encoded_payload(png_pixels, "png")
        raw_payload = bytes([6, 5, 4, 3, 2, 1])
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            recording, frames = _record(
                root,
                [
                    (png_payload, _frame(png_payload, sequence=4)),
                    (
                        raw_payload,
                        _frame(
                            raw_payload,
                            payload_format="raw",
                            pixel_format="bgr8",
                            sequence=5,
                        ),
                    ),
                ],
            )
            report = validate_decode_determinism(frames, recording, repetitions=2)
            decoded = list(CameraReplaySource(recording).iter_decoded())
        self.assertTrue(report["deterministic"])
        self.assertEqual(report["frame_count"], 2)
        self.assertEqual(
            [result.image.sequence_number for result in decoded],
            [4, 5],
        )
        self.assertEqual(
            report["decoded_content_sha256"],
            [result.image.decoded_content_sha256 for result in decoded],
        )

    def test_load_and_decode_timings_are_measured_without_later_stages(self):
        payload = _encoded_payload(
            np.array([[[1, 2, 3], [4, 5, 6]]], dtype=np.uint8), "png"
        )
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            recording, frames = _record(root, [(payload, _frame(payload))])
            result = decode_camera_payload(frames[0], recording)
        self.assertGreaterEqual(result.timing.payload_load_ms, 0.0)
        self.assertGreaterEqual(result.timing.decode_ms, 0.0)
        self.assertIsNone(result.timing.preprocess_ms)
        self.assertIsNone(result.timing.inference_ms)
        summary = summarize_visual_timings(
            [result.timing, VisualTiming(payload_load_ms=None, decode_ms=None)]
        )
        self.assertEqual(summary["stages"]["payload_load_ms"]["count"], 1)
        self.assertEqual(summary["stages"]["decode_ms"]["count"], 1)
        self.assertEqual(summary["stages"]["preprocess_ms"]["count"], 0)


if __name__ == "__main__":
    unittest.main()

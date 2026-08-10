import asyncio
from contextlib import redirect_stdout
import hashlib
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from src.cli import sensors
from src.sensors.camera_recording import (
    CameraManifestError,
    CameraPayloadCorruptedError,
    CameraPayloadLayoutError,
    CameraPayloadMissingError,
    CameraReplaySource,
    CameraReplayTimestampError,
    UnsupportedCameraRecordingSchema,
    record_camera_frames,
    record_camera_frames_async,
    validate_camera_recording,
)
from src.sensors.types import CameraFrame


def _sha256(value):
    return hashlib.sha256(value).hexdigest()


def _frame(payload, *, sequence=0, timestamp=1.0, path="source.png"):
    return CameraFrame(
        frame_id=f"frame-{sequence}",
        source_id="synthetic_camera",
        sequence_number=sequence,
        capture_timestamp=timestamp,
        capture_clock_domain="simulator",
        receive_monotonic_timestamp=10.0 + sequence,
        width=2,
        height=1,
        payload_format="png",
        pixel_format="rgb8",
        payload_relative_path=path,
        payload_sha256=_sha256(payload),
        metadata={"exposure": 0.25, "valid": True},
    )


class CameraFrameContractTests(unittest.TestCase):
    def test_valid_frame_round_trips_without_payload_bytes(self):
        payload = b"tiny-png"
        frame = _frame(payload)
        self.assertEqual(CameraFrame.from_record(frame.to_record()), frame)
        self.assertNotIn(payload.decode(), json.dumps(frame.to_record()))

    def test_invalid_dimensions_are_rejected(self):
        for field in ("width", "height"):
            values = _frame(b"x").to_record()
            values[field] = 0
            with self.assertRaisesRegex(ValueError, field):
                CameraFrame.from_record(values)

    def test_invalid_timestamps_formats_path_and_hash_are_rejected(self):
        cases = {
            "capture_timestamp": float("nan"),
            "receive_monotonic_timestamp": float("inf"),
            "payload_format": "unsupported",
            "pixel_format": "unsupported",
            "payload_relative_path": "/absolute/frame.png",
            "payload_sha256": "not-a-hash",
        }
        for field, value in cases.items():
            with self.subTest(field=field):
                values = _frame(b"x").to_record()
                values[field] = value
                with self.assertRaises(ValueError):
                    CameraFrame.from_record(values)

    def test_metadata_must_be_json_serializable(self):
        values = _frame(b"x").to_record()
        values["metadata"] = {"not_json": object()}
        with self.assertRaisesRegex(ValueError, "JSON-serializable"):
            CameraFrame.from_record(values)


class CameraRecordingTests(unittest.TestCase):
    def test_recording_layout_hashes_order_and_health_summary(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            payload = b"synthetic-image-payload"
            (root / "source.png").write_bytes(payload)
            frames = [
                _frame(payload, sequence=1, timestamp=1.0),
                _frame(payload, sequence=3, timestamp=3.0),
                _frame(payload, sequence=3, timestamp=2.0),
            ]
            recording = root / "recording"
            summary = record_camera_frames(frames, recording, payload_root=root)
            self.assertTrue((recording / "metadata.json").is_file())
            self.assertTrue((recording / "frames.jsonl").is_file())
            self.assertTrue((recording / "summary.json").is_file())
            self.assertEqual(
                sorted(path.name for path in (recording / "frames").iterdir()),
                ["000000000.png", "000000001.png", "000000002.png"],
            )
            rows = [
                json.loads(line)
                for line in (recording / "frames.jsonl").read_text().splitlines()
            ]
            self.assertEqual([row["sequence_number"] for row in rows], [1, 3, 3])
            self.assertTrue(all(row["payload_format"] == "png" for row in rows))
            self.assertTrue(all(row["pixel_format"] == "rgb8" for row in rows))
            self.assertTrue(all("encoding" not in row for row in rows))
            self.assertEqual(
                [row["payload_relative_path"] for row in rows],
                [
                    "frames/000000000.png",
                    "frames/000000001.png",
                    "frames/000000002.png",
                ],
            )
            self.assertTrue(
                all(row["payload_sha256"] == _sha256(payload) for row in rows)
            )
            self.assertNotIn(payload.decode(), (recording / "frames.jsonl").read_text())
            self.assertEqual(summary["missing_sequence_count"], 1)
            self.assertEqual(summary["duplicate_sequence_count"], 1)
            self.assertEqual(summary["non_monotonic_timestamp_count"], 1)
            self.assertIsNone(summary["timestamp_span_s"])
            self.assertEqual(summary["payload_formats"], ["png"])
            self.assertEqual(summary["pixel_formats"], ["rgb8"])

    def test_raw_payload_requires_tightly_packed_uint8_without_padding(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            valid_payload = bytes(range(6))
            (root / "valid.raw").write_bytes(valid_payload)
            raw_frame = CameraFrame(
                **{
                    **_frame(
                        valid_payload,
                        path="valid.raw",
                    ).__dict__,
                    "payload_format": "raw",
                }
            )
            summary = record_camera_frames(
                [raw_frame], root / "valid-recording", payload_root=root
            )
            self.assertEqual(summary["accepted_frame_count"], 1)
            self.assertTrue(
                (root / "valid-recording/frames/000000000.raw").is_file()
            )
            invalid_payload = b"padding"
            (root / "invalid.raw").write_bytes(invalid_payload)
            invalid_frame = CameraFrame(
                **{
                    **_frame(
                        invalid_payload,
                        path="invalid.raw",
                    ).__dict__,
                    "payload_format": "raw",
                }
            )
            invalid_summary = record_camera_frames(
                [invalid_frame], root / "invalid-recording", payload_root=root
            )
            self.assertEqual(invalid_summary["accepted_frame_count"], 0)
            self.assertEqual(
                invalid_summary["invalid_reason_counts"],
                {"CameraPayloadLayoutError": 1},
            )

    def test_invalid_input_is_counted_with_an_explicit_reason(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            payload = b"x"
            (root / "source.png").write_bytes(payload)
            invalid = _frame(payload).to_record()
            invalid["width"] = 0
            summary = record_camera_frames(
                [_frame(payload), invalid], root / "recording", payload_root=root
            )
            self.assertEqual(summary["accepted_frame_count"], 1)
            self.assertEqual(summary["invalid_frame_count"], 1)
            metadata = json.loads((root / "recording/metadata.json").read_text())
            self.assertEqual(metadata["invalid_frames"][0]["reason"], "ValueError")
            self.assertIn("width", metadata["invalid_frames"][0]["message"])

    def test_async_iterable_uses_the_same_layout(self):
        async def frames(frame):
            yield frame

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            payload = b"async"
            (root / "source.png").write_bytes(payload)
            summary = asyncio.run(
                record_camera_frames_async(
                    frames(_frame(payload)),
                    root / "recording",
                    payload_root=root,
                )
            )
            self.assertEqual(summary["accepted_frame_count"], 1)


class CameraReplayTests(unittest.TestCase):
    def _recording(self, root):
        payloads = (b"first", b"second")
        for index, payload in enumerate(payloads):
            (root / f"source-{index}.png").write_bytes(payload)
        frames = [
            _frame(
                payload,
                sequence=index,
                timestamp=float(index),
                path=f"source-{index}.png",
            )
            for index, payload in enumerate(payloads)
        ]
        recording = root / "recording"
        record_camera_frames(frames, recording, payload_root=root)
        return recording

    def test_two_no_sleep_replays_are_identical_and_manifest_ordered(self):
        with tempfile.TemporaryDirectory() as directory:
            recording = self._recording(Path(directory))
            first_source = CameraReplaySource(recording)
            second_source = CameraReplaySource(recording)
            first = [frame.to_record() for frame in first_source.iter_frames()]
            second = [frame.to_record() for frame in second_source.iter_frames()]
            self.assertEqual(first, second)
            self.assertEqual([row["sequence_number"] for row in first], [0, 1])
            self.assertEqual(first_source.replay_summary(), second_source.replay_summary())
            self.assertEqual(
                first_source.replay_summary()["ordering_rule"],
                "frames.jsonl manifest order",
            )

    def test_missing_and_corrupted_payloads_raise_typed_errors(self):
        with tempfile.TemporaryDirectory() as directory:
            recording = self._recording(Path(directory))
            (recording / "frames/000000000.png").unlink()
            with self.assertRaises(CameraPayloadMissingError):
                list(CameraReplaySource(recording).iter_frames())
        with tempfile.TemporaryDirectory() as directory:
            recording = self._recording(Path(directory))
            (recording / "frames/000000000.png").write_bytes(b"corrupt")
            with self.assertRaises(CameraPayloadCorruptedError):
                list(CameraReplaySource(recording).iter_frames())

    def test_replay_rejects_raw_payload_with_invalid_layout(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            payload = bytes(range(6))
            (root / "source.raw").write_bytes(payload)
            frame = CameraFrame(
                **{
                    **_frame(payload, path="source.raw").__dict__,
                    "payload_format": "raw",
                }
            )
            recording = root / "recording"
            record_camera_frames([frame], recording, payload_root=root)
            payload_path = recording / "frames/000000000.raw"
            corrupted = payload + b"padding"
            payload_path.write_bytes(corrupted)
            manifest_path = recording / "frames.jsonl"
            row = json.loads(manifest_path.read_text())
            row["payload_sha256"] = _sha256(corrupted)
            manifest_path.write_text(json.dumps(row) + "\n")
            with self.assertRaises(CameraPayloadLayoutError):
                list(CameraReplaySource(recording).iter_frames())

    def test_malformed_manifest_and_unsupported_schema_raise_typed_errors(self):
        with tempfile.TemporaryDirectory() as directory:
            recording = self._recording(Path(directory))
            (recording / "frames.jsonl").write_text("{broken\n")
            with self.assertRaises(CameraManifestError):
                validate_camera_recording(recording)
        with tempfile.TemporaryDirectory() as directory:
            recording = self._recording(Path(directory))
            metadata_path = recording / "metadata.json"
            metadata = json.loads(metadata_path.read_text())
            metadata["recording_schema_version"] = 999
            metadata_path.write_text(json.dumps(metadata))
            with self.assertRaises(UnsupportedCameraRecordingSchema):
                validate_camera_recording(recording)

    def test_paced_replay_rejects_non_monotonic_source_timestamps(self):
        async def consume(source):
            return [frame async for frame in source.replay(mode="paced")]

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            payload = b"x"
            (root / "source.png").write_bytes(payload)
            record_camera_frames(
                [
                    _frame(payload, sequence=0, timestamp=2.0),
                    _frame(payload, sequence=1, timestamp=1.0),
                ],
                root / "recording",
                payload_root=root,
            )
            with self.assertRaises(CameraReplayTimestampError):
                asyncio.run(consume(CameraReplaySource(root / "recording")))


class CameraCliTests(unittest.TestCase):
    def _recording(self, root):
        payload = b"cli"
        (root / "source.png").write_bytes(payload)
        record_camera_frames(
            [_frame(payload)], root / "recording", payload_root=root
        )
        return root / "recording"

    def test_camera_commands_are_registered_and_help_is_offline(self):
        help_text = sensors.build_parser().format_help()
        for command in ("camera-inspect", "camera-validate", "camera-replay"):
            self.assertIn(command, help_text)

    @patch("src.cli.sensors.build_lidar_source")
    def test_valid_replay_succeeds_without_starting_simulator(self, lidar_factory):
        with tempfile.TemporaryDirectory() as directory:
            recording = self._recording(Path(directory))
            with redirect_stdout(io.StringIO()):
                status = sensors.main(
                    ["camera-replay", "--input", str(recording), "--mode", "no_sleep"]
                )
        self.assertEqual(status, 0)
        lidar_factory.assert_not_called()

    def test_corrupted_recording_returns_nonzero(self):
        with tempfile.TemporaryDirectory() as directory:
            recording = self._recording(Path(directory))
            (recording / "frames/000000000.png").write_bytes(b"corrupt")
            with redirect_stdout(io.StringIO()):
                status = sensors.main(["camera-validate", "--input", str(recording)])
        self.assertNotEqual(status, 0)


if __name__ == "__main__":
    unittest.main()

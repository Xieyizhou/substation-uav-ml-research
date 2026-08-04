import base64
import asyncio
import contextlib
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest import mock
import xml.etree.ElementTree as ET

from scripts.maps.generate_test_maps import (
    GAZEBO_VISUAL_LABELS,
    MAP_SPECS,
    build_world,
)
from src.cli import visual
from src.vision.collection.gazebo_truth import (
    parse_gazebo_truth_message,
)
from src.vision.collection.pilot import (
    PilotRecordingError,
    load_pilot_protocol,
    write_pilot_recording,
)
from src.vision.collection.pilot_validation import (
    inspect_pilot_recording,
    materialize_pilot_dataset_identity,
    validate_pilot_recording,
)
from src.vision.collection.pilot_metrics import summarize_source_health


PROJECT_ROOT = Path(__file__).resolve().parents[1]
from src.vision.collection.pilot_acceptance import invalid_truth_acceptance_failures
from src.vision.collection.pilot_live import prepare_pilot_output_directory
from src.vision.collection.synchronization import synchronize_visual_frames
from src.sensors.gazebo_camera import GazeboCameraSource, parse_gazebo_image_message
from src.sensors.gazebo_visual_transport import (
    GAZEBO_SIM_CLOCK,
    GAZEBO_JSON_STREAM_LIMIT_BYTES,
    load_research_visual_configuration,
    message_timestamp,
)


def image_message(
    timestamp,
    data,
    *,
    width=2,
    height=1,
    step=None,
    pixel_format="RGB_INT8",
    sequence=None,
):
    header = {
        "stamp": {
            "sec": int(timestamp),
            "nsec": round((timestamp - int(timestamp)) * 1_000_000_000),
        },
        "data": [],
    }
    if sequence is not None:
        header["data"].append({"key": "seq", "value": [str(sequence)]})
    channels = {"L_INT8": 1, "RGB_INT8": 3, "BGR_INT8": 3}[pixel_format]
    return {
        "header": header,
        "width": width,
        "height": height,
        "step": step if step is not None else width * channels,
        "data": base64.b64encode(data).decode(),
        "pixel_format_type": pixel_format,
    }


def truth_message(timestamp, boxes, *, sequence=None):
    header = {
        "stamp": {
            "sec": int(timestamp),
            "nsec": round((timestamp - int(timestamp)) * 1_000_000_000),
        },
        "data": [],
    }
    if sequence is not None:
        header["data"].append({"key": "sequence", "value": [str(sequence)]})
    return {"header": header, "annotated_box": boxes}


def box(label, x1, y1, x2, y2):
    return {
        "label": label,
        "box": {
            "min_corner": {"x": x1, "y": y1},
            "max_corner": {"x": x2, "y": y2},
        },
    }


class GazeboConfigurationTests(unittest.TestCase):
    def test_v3_accepts_observed_boundary_event_but_rejects_larger_failures(self):
        recording = load_pilot_protocol()["recording"]
        observed = ["exact"] * 2851 + ["invalid_truth"] * 2
        self.assertEqual(
            invalid_truth_acceptance_failures(observed, 2853, recording),
            [],
        )
        excessive_fraction = ["exact"] * 998 + ["invalid_truth"] * 2
        self.assertTrue(
            invalid_truth_acceptance_failures(
                excessive_fraction, 1000, recording
            )
        )
        excessive_run = ["exact"] * 3000 + ["invalid_truth"] * 3
        failures = invalid_truth_acceptance_failures(
            excessive_run, len(excessive_run), recording
        )
        self.assertTrue(
            any("consecutive" in failure for failure in failures)
        )

    def test_research_model_declares_distinct_rgb_and_truth_rates(self):
        config = load_research_visual_configuration()
        self.assertEqual(config["rgb"]["configured_topic"], "research_camera/image")
        self.assertEqual(config["truth"]["configured_topic"], "research_camera/boxes")
        self.assertEqual(config["rgb"]["configured_update_rate_hz"], 30.0)
        self.assertEqual(config["truth"]["configured_update_rate_hz"], 15.0)
        self.assertEqual(config["truth"]["box_type"], "full_2d")
        self.assertEqual(config["rgb"]["width"], config["truth"]["width"])

    def test_research_model_uses_lightweight_lidar_payload(self):
        model = ET.parse(
            PROJECT_ROOT / "simulation/models/x500_research/model.sdf"
        ).getroot()
        lidar_link = model.find(".//link[@name='research_lidar_link']")
        self.assertIsNotNone(lidar_link)
        self.assertLessEqual(
            float(lidar_link.findtext("inertial/mass")),
            0.05,
        )
        self.assertIsNotNone(
            lidar_link.find("sensor[@name='lidar_2d_v2']")
        )

    def test_generated_equipment_uses_locked_simulator_labels(self):
        self.assertEqual(
            GAZEBO_VISUAL_LABELS,
            {
                "transformer": 1,
                "switchgear": 2,
                "capacitor_bank": 3,
                "reactor": 4,
            },
        )
        tree, _ = build_world(MAP_SPECS[2])
        models = {
            model.get("name"): model for model in tree.getroot().findall(".//model")
        }
        for obstacle in MAP_SPECS[2]["obstacles"]:
            label = models[obstacle["name"]].findtext(
                "plugin[@name='gz::sim::systems::Label']/label"
            )
            expected = GAZEBO_VISUAL_LABELS.get(obstacle["visual_category"])
            self.assertEqual(int(label) if label is not None else None, expected)

    def test_classic_simple_world_labels_only_transformers(self):
        world = ET.parse(
            PROJECT_ROOT / "simulation/worlds/substation_simple.sdf"
        ).getroot()
        models = {
            model.get("name"): model
            for model in world.findall(".//model")
        }
        for name in ("transformer_1", "transformer_2"):
            self.assertEqual(
                models[name].findtext(
                    "plugin[@name='gz::sim::systems::Label']/label"
                ),
                "1",
            )
        for name in (
            "switchgear_1",
            "switchgear_2",
            "capacitor_bank",
            "reactor",
            "control_building",
        ):
            if name in models:
                self.assertIsNone(
                    models[name].find(
                        "plugin[@name='gz::sim::systems::Label']"
                    )
                )

    def test_timestamp_requires_gazebo_header(self):
        with self.assertRaisesRegex(ValueError, "header"):
            message_timestamp({})

    def test_protocol_v3_retains_frames_and_bounds_invalid_truth(self):
        protocol = load_pilot_protocol()
        self.assertEqual(protocol["protocol_version"], 3)
        self.assertEqual(
            protocol["recording"]["expected_source_rate_hz"], 10.0
        )
        self.assertEqual(
            protocol["recording"][
                "default_rate_observation_tolerance_fraction"
            ],
            0.1,
        )
        self.assertTrue(
            protocol["recording"]["retain_every_valid_source_frame"]
        )
        self.assertFalse(protocol["recording"]["automatic_downsampling"])
        self.assertEqual(
            protocol["recording"]["maximum_invalid_truth_rgb_fraction"],
            0.001,
        )
        self.assertEqual(
            protocol["recording"][
                "maximum_consecutive_invalid_truth_rgb_frames"
            ],
            2,
        )


class GazeboRgbConversionTests(unittest.TestCase):
    def test_rgb_and_bgr_are_converted_explicitly_to_png(self):
        with tempfile.TemporaryDirectory() as directory:
            rgb = parse_gazebo_image_message(
                image_message(1.25, bytes([255, 0, 0, 0, 255, 0]), sequence=9),
                topic="/research_camera/image",
                staging_directory=directory,
                receive_index=1,
                received_monotonic=5.0,
            )
            bgr = parse_gazebo_image_message(
                image_message(
                    1.5,
                    bytes([0, 0, 255, 0, 255, 0]),
                    pixel_format="BGR_INT8",
                ),
                topic="/research_camera/image",
                staging_directory=directory,
                receive_index=2,
                received_monotonic=6.0,
            )
            self.assertEqual(rgb.payload_format, "png")
            self.assertEqual(rgb.pixel_format, "rgb8")
            self.assertEqual(rgb.sequence_number, 9)
            self.assertEqual(rgb.metadata["sequence_provenance"], "gazebo_header")
            self.assertEqual(bgr.sequence_number, 2)
            self.assertEqual(bgr.metadata["source_pixel_format"], "bgr8")
            self.assertEqual(rgb.payload_sha256, bgr.payload_sha256)

    def test_row_padding_is_removed_and_provenance_is_preserved(self):
        with tempfile.TemporaryDirectory() as directory:
            frame = parse_gazebo_image_message(
                image_message(
                    2.0,
                    bytes([1, 2, 3, 4, 5, 6, 99, 99]),
                    step=8,
                ),
                topic="/research_camera/image",
                staging_directory=directory,
                receive_index=1,
            )
            self.assertEqual(frame.metadata["source_row_stride_bytes"], 8)
            self.assertEqual(frame.capture_clock_domain, GAZEBO_SIM_CLOCK)

    def test_invalid_dimensions_payload_and_format_are_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            for message in (
                image_message(1, b"", width=0),
                image_message(1, b"\0"),
                {
                    **image_message(1, b"\0" * 6),
                    "pixel_format_type": "RGB_INT16",
                },
            ):
                with self.subTest(message=message):
                    with self.assertRaises(ValueError):
                        parse_gazebo_image_message(
                            message,
                            topic="/research_camera/image",
                            staging_directory=directory,
                            receive_index=1,
                        )


class _BlockingStream:
    async def readline(self):
        await asyncio.Event().wait()

    async def read(self):
        return b""


class _FakeProcess:
    def __init__(self):
        self.stdout = _BlockingStream()
        self.stderr = _BlockingStream()
        self.returncode = None

    def terminate(self):
        self.returncode = 0

    def kill(self):
        self.returncode = -9

    async def wait(self):
        return self.returncode

    async def communicate(self):
        return b"", b""


class GazeboRgbLifecycleTests(unittest.IsolatedAsyncioTestCase):
    async def test_subprocess_stream_limit_accepts_full_hd_json_frames(self):
        process = _FakeProcess()
        with tempfile.TemporaryDirectory() as directory:
            source = GazeboCameraSource(directory, topic="/research_camera/image")
            with (
                mock.patch(
                    "src.sensors.gazebo_camera.inspect_topic",
                    new=mock.AsyncMock(),
                ),
                mock.patch(
                    "src.sensors.gazebo_camera.asyncio.create_subprocess_exec",
                    new=mock.AsyncMock(return_value=process),
                ) as spawn,
            ):
                await source.start()
        self.assertEqual(
            spawn.await_args.kwargs["limit"],
            GAZEBO_JSON_STREAM_LIMIT_BYTES,
        )

    async def test_stream_timeout_is_explicit_and_shutdown_is_clean(self):
        with tempfile.TemporaryDirectory() as directory:
            source = GazeboCameraSource(directory, topic="/research_camera/image")
            source._process = _FakeProcess()
            with self.assertRaises(TimeoutError):
                await anext(source.events(timeout_s=0.001))
            await source.stop()
            self.assertIsNone(source._process)

    async def test_stream_iteration_is_cancellable(self):
        with tempfile.TemporaryDirectory() as directory:
            source = GazeboCameraSource(directory, topic="/research_camera/image")
            source._process = _FakeProcess()
            pending = asyncio.create_task(
                anext(source.events(timeout_s=10.0))
            )
            await asyncio.sleep(0)
            pending.cancel()
            with self.assertRaises(asyncio.CancelledError):
                await pending
            await source.stop()


class GazeboTruthTests(unittest.TestCase):
    def test_protobuf_json_omitted_zero_coordinate_is_preserved(self):
        message = truth_message(1.0, [])
        message["annotated_box"] = [
            {
                "box": {
                    "minCorner": {"y": 1.0},
                    "maxCorner": {"x": 2.0, "y": 3.0},
                },
                "label": 2,
            }
        ]
        truth = parse_gazebo_truth_message(
            message,
            topic="/research_camera/boxes",
            width=4,
            height=3,
            receive_index=1,
        )
        self.assertTrue(truth.valid)
        self.assertEqual(truth.objects[0].bbox_xyxy, (0.0, 1.0, 2.0, 3.0))

    def test_locked_label_mapping_clipping_and_provenance(self):
        truth = parse_gazebo_truth_message(
            truth_message(1.0, [box(1, -1, 0, 2, 1)], sequence=4),
            topic="/research_camera/boxes",
            width=4,
            height=3,
            receive_index=1,
        )
        self.assertTrue(truth.valid)
        self.assertEqual(truth.source_sequence, 4)
        self.assertEqual(truth.objects[0].class_name, "transformer")
        self.assertEqual(truth.objects[0].bbox_xyxy, (0.0, 0.0, 2.0, 1.0))
        self.assertEqual(truth.objects[0].truncation_status, "truncated")
        self.assertEqual(truth.objects[0].visibility_status, "unknown")

    def test_no_target_is_valid_but_unknown_and_zero_area_are_invalid(self):
        no_target = parse_gazebo_truth_message(
            truth_message(1.0, []),
            topic="/research_camera/boxes",
            width=4,
            height=3,
            receive_index=1,
        )
        invalid = parse_gazebo_truth_message(
            truth_message(
                1.0,
                [box(99, 0, 0, 1, 1), box(1, 2, 2, 2, 3)],
            ),
            topic="/research_camera/boxes",
            width=4,
            height=3,
            receive_index=2,
        )
        self.assertTrue(no_target.valid)
        self.assertEqual(no_target.objects, ())
        self.assertFalse(invalid.valid)
        self.assertEqual(len(invalid.invalid_reasons), 2)


class SynchronizationTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.frames = [
            parse_gazebo_image_message(
                image_message(timestamp, bytes([255, 0, 0, 0, 255, 0])),
                topic="/research_camera/image",
                staging_directory=self.temporary.name,
                receive_index=index,
            )
            for index, timestamp in enumerate((1.0, 1.03, 2.0), start=1)
        ]

    def truth(self, timestamp, receive_index, boxes=()):
        return parse_gazebo_truth_message(
            truth_message(timestamp, list(boxes)),
            topic="/research_camera/boxes",
            width=2,
            height=1,
            receive_index=receive_index,
        )

    def test_exact_nearest_and_tolerance_rejection(self):
        result = synchronize_visual_frames(
            self.frames,
            [self.truth(1.0, 1), self.truth(1.05, 2)],
            maximum_skew_ms=25.0,
        )
        self.assertEqual(result[0].synchronization.synchronization_status, "exact")
        self.assertEqual(
            result[1].synchronization.synchronization_status,
            "nearest_within_tolerance",
        )
        self.assertEqual(
            result[2].synchronization.synchronization_status, "unmatched"
        )

    def test_tie_is_ambiguous_and_deterministically_references_earlier(self):
        result = synchronize_visual_frames(
            [self.frames[1]],
            [self.truth(1.01, 2), self.truth(1.05, 1)],
            maximum_skew_ms=25.0,
        )[0]
        self.assertEqual(result.synchronization.synchronization_status, "ambiguous")
        self.assertIsNone(result.truth)
        self.assertEqual(result.synchronization.truth_simulation_timestamp, 1.01)

    def test_large_truth_stream_uses_only_nearest_timestamp_candidates(self):
        truths = [
            self.truth(index / 100.0, index + 1)
            for index in range(2000)
        ]
        with mock.patch(
            "src.vision.collection.synchronization._candidate_key",
            wraps=lambda frame, truth: (
                abs(truth.simulation_timestamp - frame.capture_timestamp),
                truth.simulation_timestamp,
                truth.message_id,
            ),
        ) as candidate_key:
            result = synchronize_visual_frames(self.frames, truths)
        self.assertEqual(len(result), len(self.frames))
        self.assertLessEqual(candidate_key.call_count, len(self.frames) * 2)

    def test_invalid_truth_is_not_no_target(self):
        invalid = parse_gazebo_truth_message(
            truth_message(1.0, [box(99, 0, 0, 1, 1)]),
            topic="/research_camera/boxes",
            width=2,
            height=1,
            receive_index=1,
        )
        result = synchronize_visual_frames([self.frames[0]], [invalid])[0]
        self.assertEqual(
            result.synchronization.synchronization_status, "invalid_truth"
        )
        self.assertIsNone(result.truth)


class PilotRecordingTests(unittest.TestCase):
    def build_valid_pilot(self, directory, *, invalid_truth_index=None):
        pixels = bytes([255, 0, 0, 0, 255, 0])
        timestamps = tuple(index + 0.1 for index in range(10))
        frames = [
            parse_gazebo_image_message(
                image_message(timestamp, pixels, sequence=index),
                topic="/research_camera/image",
                staging_directory=directory,
                receive_index=index,
            )
            for index, timestamp in enumerate(timestamps, start=1)
        ]
        truths = [
            parse_gazebo_truth_message(
                truth_message(
                    timestamp,
                    (
                        [box(99, 0, 0, 1, 1)]
                        if index == invalid_truth_index
                        else [] if index == 10 else [box(2, 0, 0, 1, 1)]
                    ),
                ),
                topic="/research_camera/boxes",
                width=2,
                height=1,
                receive_index=index,
            )
            for index, timestamp in enumerate(timestamps, start=1)
        ]
        events = [
            {"simulation_timestamp": 0.0, "mission_phase": "cruise_distant"},
            {"simulation_timestamp": 3.0, "mission_phase": "approach"},
            {"simulation_timestamp": 5.0, "mission_phase": "close_inspection"},
            {"simulation_timestamp": 8.0, "mission_phase": "target_transition"},
        ]
        return write_pilot_recording(
            directory,
            frames,
            truths,
            recording_id="pilot-test",
            mission_events=events,
        )

    def test_complete_pilot_writes_all_manifests_and_identity(self):
        with tempfile.TemporaryDirectory() as directory:
            summary = self.build_valid_pilot(directory)
            self.assertEqual(summary["recording_state"], "complete")
            self.assertIn(
                "observed_rate_outside_configured_tolerance",
                summary["source_health"]["observation_flags"],
            )
            validation = validate_pilot_recording(directory)
            self.assertEqual(validation["frame_count"], 10)
            self.assertEqual(validation["source_frame_count"], 10)
            self.assertGreaterEqual(
                validation["mission_phase_duration_s"]["close_inspection"],
                1.0,
            )
            identity = materialize_pilot_dataset_identity(directory)
            self.assertEqual(identity.dataset_role, "pilot")
            self.assertEqual(identity.source_payload_formats, ("png",))
            self.assertEqual(identity.labelled_frame_count, 9)
            self.assertTrue(
                Path(directory, "identity/dataset_identity.json").is_file()
            )
            inspection = inspect_pilot_recording(directory)
            self.assertTrue(inspection["dataset_identity_exists"])
            self.assertEqual(
                inspection["synchronization"]["no_target_frame_count"], 1
            )
            self.assertTrue(Path(directory, "truth_events.jsonl").is_file())

    def test_invalid_truth_is_retained_but_only_valid_frames_enter_identity(self):
        with tempfile.TemporaryDirectory() as directory:
            self.build_valid_pilot(directory, invalid_truth_index=2)
            with self.assertRaisesRegex(
                PilotRecordingError, "invalid-truth RGB fraction"
            ):
                validate_pilot_recording(directory)
            protocol = load_pilot_protocol()
            protocol["recording"]["maximum_invalid_truth_rgb_fraction"] = 0.2
            with mock.patch(
                "src.vision.collection.pilot_validation.load_pilot_protocol",
                return_value=protocol,
            ):
                validation = validate_pilot_recording(directory)
                identity = materialize_pilot_dataset_identity(directory)
            self.assertEqual(validation["source_frame_count"], 10)
            self.assertEqual(validation["frame_count"], 9)
            self.assertEqual(
                validation["excluded_invalid_truth_frame_count"], 1
            )
            self.assertEqual(identity.ordered_frame_count, 9)
            self.assertEqual(identity.labelled_frame_count, 8)
            self.assertEqual(
                len(
                    Path(directory, "identity/dataset_membership.jsonl")
                    .read_text()
                    .splitlines()
                ),
                9,
            )
            truth_events = [
                json.loads(line)
                for line in Path(directory, "truth_events.jsonl")
                .read_text()
                .splitlines()
            ]
            invalid = [
                event
                for event in truth_events
                if event["validation_status"] == "invalid"
            ]
            self.assertEqual(len(invalid), 1)
            self.assertIn("unknown simulator label", invalid[0]["invalid_reasons"][0])

    def test_partial_or_missing_manifest_cannot_materialize(self):
        with tempfile.TemporaryDirectory() as directory:
            self.build_valid_pilot(directory)
            metadata_path = Path(directory, "metadata.json")
            metadata = json.loads(metadata_path.read_text())
            metadata["recording_state"] = "in_progress"
            metadata_path.write_text(json.dumps(metadata))
            with self.assertRaises(PilotRecordingError):
                materialize_pilot_dataset_identity(directory)
        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaisesRegex(PilotRecordingError, "missing"):
                validate_pilot_recording(directory)

    def test_source_metrics_do_not_fabricate_rate(self):
        with tempfile.TemporaryDirectory() as directory:
            frame = parse_gazebo_image_message(
                image_message(1.0, bytes([255, 0, 0, 0, 255, 0])),
                topic="/research_camera/image",
                staging_directory=directory,
                receive_index=1,
            )
            summary = summarize_source_health(
                [frame], expected_source_rate_hz=10.0
            )
            self.assertIsNone(summary["effective_source_rate_hz"])
            self.assertIsNone(summary["observed_duration_s"])

    def test_rate_interval_gap_duplicate_and_stall_metrics(self):
        with tempfile.TemporaryDirectory() as directory:
            frames = [
                parse_gazebo_image_message(
                    image_message(timestamp, bytes([255, 0, 0, 0, 255, 0]), sequence=sequence),
                    topic="/research_camera/image",
                    staging_directory=directory,
                    receive_index=index,
                    received_monotonic=receive,
                )
                for index, (timestamp, sequence, receive) in enumerate(
                    ((0.0, 1, 0.0), (0.1, 3, 0.1), (0.2, 3, 1.0)),
                    start=1,
                )
            ]
            summary = summarize_source_health(
                frames,
                expected_source_rate_hz=20.0,
                expected_rate_tolerance_fraction=0.1,
                jitter_p95_limit_ms=90.0,
                receive_stall_limit_ms=500.0,
            )
            self.assertAlmostEqual(summary["effective_source_rate_hz"], 10.0)
            self.assertAlmostEqual(
                summary["inter_frame_interval_ms"]["p95"], 100.0
            )
            self.assertEqual(summary["sequence_gap_count"], 1)
            self.assertEqual(summary["duplicate_sequence_count"], 1)
            self.assertEqual(summary["receive_time_stall_count"], 1)
            self.assertEqual(
                summary["observation_flags"],
                [
                    "excessive_jitter",
                    "observed_rate_outside_configured_tolerance",
                    "receive_time_stalls",
                    "sequence_gaps",
                ],
            )

    def test_non_monotonic_timestamp_suppresses_observed_rate(self):
        with tempfile.TemporaryDirectory() as directory:
            frames = [
                parse_gazebo_image_message(
                    image_message(timestamp, bytes([255, 0, 0, 0, 255, 0])),
                    topic="/research_camera/image",
                    staging_directory=directory,
                    receive_index=index,
                )
                for index, timestamp in enumerate((1.0, 0.9), start=1)
            ]
            summary = summarize_source_health(
                frames, expected_source_rate_hz=10.0
            )
            self.assertIsNone(summary["effective_source_rate_hz"])
            self.assertEqual(
                summary["non_monotonic_simulation_timestamp_count"], 1
            )
            self.assertIn(
                "timestamp_health_failure", summary["observation_flags"]
            )


class VisualPilotCliTests(unittest.TestCase):
    def test_recording_output_directory_must_be_empty(self):
        with tempfile.TemporaryDirectory() as directory:
            empty_output = Path(directory, "new-recording")
            self.assertEqual(
                prepare_pilot_output_directory(empty_output), empty_output
            )
            Path(empty_output, "live_status.json").write_text("{}")
            with self.assertRaisesRegex(ValueError, "not empty"):
                prepare_pilot_output_directory(empty_output)

    def test_recording_requires_a_limit_or_explicit_interrupt_mode(self):
        parser = visual.build_parser()
        with self.assertRaises(SystemExit):
            parser.parse_args(
                ["pilot-record", "--output", "pilot-without-stop"]
            )
        for option in (
            ("--duration", "20"),
            ("--frame-limit", "300"),
            ("--until-interrupt",),
        ):
            with self.subTest(option=option):
                arguments = parser.parse_args(
                    ["pilot-record", "--output", "pilot", *option]
                )
                self.assertEqual(arguments.command, "pilot-record")

    def test_help_is_offline_and_lists_pilot_commands(self):
        output = visual.build_parser().format_help()
        self.assertIn("pilot-record", output)
        self.assertIn("pilot-phase", output)
        self.assertIn("gazebo-probe", output)

    def test_missing_live_source_returns_nonzero(self):
        with mock.patch(
            "src.cli.visual.inspect_visual_sources",
            new=mock.AsyncMock(side_effect=RuntimeError("missing configured topic")),
        ):
            output = io.StringIO()
            with contextlib.redirect_stdout(output):
                result = visual.main(["gazebo-inspect", "--timeout", "0.01"])
        self.assertEqual(result, 1)
        self.assertIn("missing configured topic", output.getvalue())

    def test_offline_pilot_inspect_validate_and_materialize(self):
        with tempfile.TemporaryDirectory() as directory:
            PilotRecordingTests().build_valid_pilot(directory)
            with mock.patch("asyncio.create_subprocess_exec") as subprocess:
                for command in ("pilot-inspect", "pilot-validate"):
                    output = io.StringIO()
                    with contextlib.redirect_stdout(output):
                        result = visual.main([command, "--input", directory])
                    self.assertEqual(result, 0)
                with contextlib.redirect_stdout(io.StringIO()):
                    result = visual.main(
                        ["pilot-materialize", "--input", directory]
                    )
                self.assertEqual(result, 0)
                subprocess.assert_not_called()

    def test_incomplete_pilot_validation_returns_nonzero(self):
        with tempfile.TemporaryDirectory() as directory:
            output = io.StringIO()
            with contextlib.redirect_stdout(output):
                result = visual.main(
                    ["pilot-validate", "--input", directory]
                )
        self.assertEqual(result, 1)
        self.assertIn("missing required pilot manifest", output.getvalue())

    def test_live_phase_marker_uses_last_rgb_simulation_timestamp(self):
        with tempfile.TemporaryDirectory() as directory:
            Path(directory, "live_status.json").write_text(
                json.dumps(
                    {
                        "recording_id": "pilot-live",
                        "accepted_frame_count": 3,
                        "last_simulation_timestamp": 12.5,
                    }
                )
            )
            with contextlib.redirect_stdout(io.StringIO()):
                result = visual.main(
                    [
                        "pilot-phase",
                        "--output",
                        directory,
                        "--phase",
                        "approach",
                    ]
                )
            self.assertEqual(result, 0)
            event = json.loads(
                Path(directory, "mission_events.jsonl").read_text()
            )
            self.assertEqual(event["simulation_timestamp"], 12.5)
            self.assertEqual(event["mission_phase"], "approach")


if __name__ == "__main__":
    unittest.main()

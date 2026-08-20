import json
from pathlib import Path
import tempfile
import time
import unittest
import xml.etree.ElementTree as ET

from src.ml.artifacts import file_sha256
from src.ml.dataset import ResearchSample
from src.ml.dataset_builder import (
    build_dataset_manifest,
    collect_replay,
    validate_dataset_directory,
)
from src.ml.domain_randomization import (
    load_ranges,
    materialize_planner_config,
    materialize_world,
    sample_manifest,
)
from src.ml.model_package import create_model_package, validate_model_package
from src.ml.research_recorder import ResearchDatasetWriter
from src.ml.scenarios import formal_scenarios, split_for
from src.ml.truth_labels import label_scan, recommended_direction_deg
from src.sensors.fault_injection import FaultInjectedLidarSource
from src.sensors.replay import append_scan_record
from src.sensors.types import LaserScanFrame, SensorHealth
from src.study.comparison import (
    bootstrap_interval,
    paired_differences,
    promotion_gate,
    formal_comparison_report,
    study_gate_report,
)
from src.study.matrix import tier_matrix
from src.study.registry import ResearchRegistry
from src.study.runner import write_run_queue
from src.study.capability_gate import capability_coverage_report
from src.study.capability_scenario import (
    apply_scenario_profile,
    route_blocker_specification,
)
from src.study.challenge_spec import challenge_matrix, load_challenge_spec


ROOT = Path(__file__).resolve().parents[1]


def research_sample(split, map_id, seed, scenario):
    return ResearchSample(
        scenario_id=scenario,
        split=split,
        map_id=map_id,
        seed=seed,
        timestamp_s=1.0,
        ranges_m=(1.0, 2.0, 4.0),
        range_max_m=10.0,
        velocity_ned_m_s=(0.5, 0.0, 0.0),
        pose_ned_m=(0.0, 0.0, -1.5),
        yaw_deg=0.0,
        risk_label="warning",
        traversability=(0.0,) * 72,
        ground_truth_occupancy=(1.0,) * 72,
        recommended_direction_deg=0.0,
    )


class ScenarioAndTruthTests(unittest.TestCase):
    def test_versioned_split_policy_reserves_formal_seeds_and_extreme(self):
        self.assertEqual(split_for("complex", 2001), "train")
        self.assertEqual(split_for("simple", 2041), "validation")
        self.assertEqual(split_for("extreme", 2051), "test")
        with self.assertRaisesRegex(ValueError, "formal evaluation"):
            split_for("simple", 1001)
        with self.assertRaisesRegex(ValueError, "held-out extreme"):
            split_for("extreme", 2001)

    def test_formal_plan_is_30_unique_balanced_scenarios(self):
        scenarios = formal_scenarios()
        self.assertEqual(len(scenarios), 30)
        self.assertEqual(len({row["scenario_id"] for row in scenarios}), 30)
        self.assertEqual(len(tier_matrix("formal")), 120)
        self.assertEqual(len(tier_matrix("closed-loop")), 15)
        self.assertEqual(len(tier_matrix("speed-envelope")), 60)

    def test_challenge_gate_is_three_runs_on_one_grounded_blocker(self):
        specification = load_challenge_spec()
        matrix = challenge_matrix()
        self.assertEqual(len(matrix), 3)
        self.assertEqual({row["scenario_id"] for row in matrix}, {
            "challenge-route-blocker-1101"
        })
        self.assertEqual(
            {row["condition"] for row in matrix},
            set(specification["conditions"]),
        )
        self.assertTrue(all(
            row["scenario_profile"] == "unmapped_route_blocker_v1"
            for row in matrix
        ))
        blocker_runs = [
            row for row in tier_matrix("closed-loop")
            if row.get("scenario_profile") == "unmapped_route_blocker_v1"
        ]
        self.assertEqual(len(blocker_runs), 3)

    def test_route_blocker_intersects_route_but_preserves_a_detour(self):
        config = json.loads((ROOT / "config/substation_obstacles.json").read_text())
        config["goal_cell"] = [16, 16]
        with tempfile.TemporaryDirectory() as directory:
            planner = Path(directory) / "planner.json"
            planner.write_text(json.dumps(config))
            blocker = route_blocker_specification(planner)
        self.assertEqual(blocker["profile"], "unmapped_route_blocker_v1")
        self.assertEqual(len(blocker["grid_cell"]), 2)

    def test_route_blocker_profile_removes_uncontrolled_confounders(self):
        manifest = {
            "unknown_obstacles": [{"id": "random"}],
            "lidar_noise_stddev_m": 0.1,
            "lidar_dropout_probability": 0.2,
            "sensor_outage_probability": 0.1,
            "sensor_outage_duration_s": 1.0,
        }
        self.assertTrue(apply_scenario_profile(
            manifest, ROOT / "config/substation_obstacles.json",
            "unmapped_route_blocker_v1",
        ))
        self.assertEqual(manifest["unknown_obstacles"], [])
        self.assertFalse(manifest.get("unmapped_obstacles"))
        for name in (
            "lidar_noise_stddev_m", "lidar_dropout_probability",
            "sensor_outage_probability", "sensor_outage_duration_s",
        ):
            self.assertEqual(manifest[name], 0.0)

    def test_large_study_gate_requires_observed_functional_coverage(self):
        runs = []
        for condition in ("geometric_lidar", "ml_lidar", "geometric_ml_fusion"):
            for index in range(5):
                runs.append({
                    "condition": condition,
                    "status": "completed",
                    "metrics": {
                        "mission_success": 1,
                        "landing_success": 1,
                        "collision_count": 0,
                        "sensor_healthy_ratio": 1.0,
                        "predicted_danger_sample_count": int(index == 0),
                        "replan_attempt_count": int(index == 0),
                        "successful_replan_count": int(index == 0),
                        "active_replan_count": int(index == 0),
                    },
                })
        self.assertTrue(capability_coverage_report(runs)["passed"])
        runs[5]["metrics"]["active_replan_count"] = 0
        report = capability_coverage_report(runs)
        self.assertFalse(report["passed"])
        self.assertIn(
            "ml_lidar: active_route_replacement was not demonstrated",
            report["reasons"],
        )

    def test_capability_gate_uses_tier_specific_matrix_size(self):
        runs = []
        for condition in ("geometric_lidar", "ml_lidar", "geometric_ml_fusion"):
            runs.append({
                "condition": condition,
                "status": "completed",
                "metrics": {
                    "mission_success": 1,
                    "landing_success": 1,
                    "collision_count": 0,
                    "sensor_healthy_ratio": 1.0,
                    "predicted_danger_sample_count": 1,
                    "truth_danger_sample_count": 0,
                    "replan_attempt_count": 1,
                    "successful_replan_count": 1,
                    "active_replan_count": 1,
                },
            })
        report = capability_coverage_report(runs, tier="challenge")
        self.assertTrue(report["passed"])
        self.assertEqual(report["tier"], "challenge")
        self.assertEqual(
            report["conditions"]["ml_lidar"]["observed"][
                "truth_danger_sample_count"
            ],
            0,
        )
        self.assertFalse(
            capability_coverage_report(runs, tier="formal")["passed"]
        )

    def test_truth_labels_include_nonconstant_recommended_direction(self):
        ranges = [6.0] * 72
        ranges[36] = 0.4
        labels = label_scan(ranges, 10.0, (1.0, 0.0, 0.0))
        self.assertEqual(labels["risk_label"], "danger")
        self.assertNotEqual(labels["recommended_direction_deg"], 0.0)
        self.assertEqual(len(labels["traversability"]), 72)
        self.assertEqual(len(labels["ground_truth_occupancy"]), 72)
        self.assertEqual(recommended_direction_deg((0.1, 1.0, 0.2)), 0.0)

    def test_randomization_materializes_reproducible_launchable_sdf(self):
        config = load_ranges(ROOT / "config/perception/domain_randomization.json")
        manifest = sample_manifest(config, map_id="training", seed=2001)
        with tempfile.TemporaryDirectory() as directory:
            first = Path(directory) / "first.sdf"
            second = Path(directory) / "second.sdf"
            materialize_world(
                ROOT / "simulation/worlds/substation_training.sdf", first, manifest
            )
            materialize_world(
                ROOT / "simulation/worlds/substation_training.sdf", second, manifest
            )
            self.assertEqual(file_sha256(first), file_sha256(second))
            self.assertIn("<world", first.read_text())
            report = materialize_world(
                ROOT / "simulation/worlds/substation_training.sdf", first, manifest
            )
            planner = Path(directory) / "planner.json"
            randomized = materialize_planner_config(
                ROOT / "config/maps/substation_training.json", planner, report
            )
            self.assertGreaterEqual(
                len(randomized["obstacles"]),
                len(json.loads((ROOT / "config/maps/substation_training.json").read_text())["obstacles"]),
            )
            self.assertEqual(
                randomized["scenario_config_hash"], manifest["config_hash"]
            )

    def test_randomized_light_color_is_clamped_to_sdf_range(self):
        config = load_ranges(ROOT / "config/perception/domain_randomization.json")
        manifest = sample_manifest(config, map_id="simple", seed=2007)
        manifest["light_intensity"] = 2.0
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "world.sdf"
            materialize_world(
                ROOT / "simulation/worlds/substation_simple.sdf",
                output,
                manifest,
            )
            diffuse = ET.parse(output).getroot().findtext(
                "./world/light[@name='sun']/diffuse"
            )
        self.assertEqual(diffuse, "1 1 1 1")


class DatasetManifestTests(unittest.TestCase):
    def test_small_replay_builds_a_valid_automatically_labelled_dataset(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            replay = root / "scan.jsonl"
            append_scan_record(
                replay,
                LaserScanFrame(
                    timestamp_s=1,
                    received_monotonic_s=1,
                    frame_id="lidar",
                    angle_min_rad=-1,
                    angle_max_rad=1,
                    angle_step_rad=1,
                    range_min_m=0.1,
                    range_max_m=10,
                    ranges_m=(5.0, 0.5, 5.0),
                    source="fixture",
                    sequence=1,
                ),
            )
            manifest = {
                "seed": 2001,
                "config_hash": "b" * 64,
                "lidar_noise_stddev_m": 0,
                "lidar_dropout_probability": 0,
                "sensor_outage_probability": 0,
                "attitude_jitter_deg": 0,
            }
            result = collect_replay(
                replay,
                root / "dataset",
                map_id="simple",
                target_id="center",
                seed=2001,
                scenario_manifest=manifest,
            )
            self.assertEqual(result["samples"], 1)
            self.assertEqual(result["risk_label_distribution"], {"warning": 1})
            self.assertEqual(
                validate_dataset_directory(root / "dataset")["dataset_id"],
                result["dataset_id"],
            )

    def test_flight_csv_is_synchronized_to_scan_receive_time(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            replay = root / "scan.jsonl"
            append_scan_record(
                replay,
                LaserScanFrame(
                    timestamp_s=100,
                    received_monotonic_s=10,
                    frame_id="lidar",
                    angle_min_rad=-1,
                    angle_max_rad=1,
                    angle_step_rad=1,
                    range_min_m=0.1,
                    range_max_m=10,
                    ranges_m=(5.0, 0.5, 5.0),
                    source="fixture",
                    sequence=1,
                ),
            )
            replay.with_suffix(".metadata.json").write_text(
                json.dumps({"wall_clock_minus_monotonic_s": 10})
            )
            telemetry = root / "flight.csv"
            telemetry.write_text(
                "timestamp_utc,local_north_m,local_east_m,local_down_m,"
                "velocity_north_m_s,velocity_east_m_s,velocity_down_m_s,"
                "yaw_deg,sensor_frame_age_s,sensor_healthy\n"
                "1970-01-01T00:00:20+00:00,1,2,-3,0.5,0.25,0,45,0.01,true\n"
            )
            manifest = {
                "seed": 2001,
                "config_hash": "b" * 64,
                "lidar_noise_stddev_m": 0,
                "lidar_dropout_probability": 0,
                "sensor_outage_probability": 0,
                "attitude_jitter_deg": 0,
            }
            collect_replay(
                replay,
                root / "dataset",
                map_id="simple",
                target_id="center",
                seed=2001,
                scenario_manifest=manifest,
                telemetry_path=telemetry,
            )
            sample = json.loads(
                (root / "dataset/samples.jsonl").read_text().splitlines()[0]
            )
            self.assertEqual(sample["pose_ned_m"], [1.0, 2.0, -3.0])
            self.assertEqual(sample["velocity_ned_m_s"], [0.5, 0.25, 0.0])
            self.assertEqual(sample["sensor_data_age_ms"], 10.0)
            self.assertTrue(sample["sensor_healthy"])

    def test_manifest_hash_detects_changed_data(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            samples = root / "samples.jsonl"
            with ResearchDatasetWriter(samples) as writer:
                writer.append(research_sample("train", "simple", 2001, "train-1"))
                writer.append(
                    research_sample("validation", "simple", 2041, "validation-1")
                )
                writer.append(research_sample("test", "extreme", 2051, "test-1"))
            manifest = build_dataset_manifest(samples)
            (root / "dataset_manifest.json").write_text(json.dumps(manifest))
            self.assertEqual(
                validate_dataset_directory(root)["dataset_id"], manifest["dataset_id"]
            )
            with samples.open("a") as output:
                output.write("\n")
            with self.assertRaisesRegex(ValueError, "data_sha256"):
                validate_dataset_directory(root)

    def test_model_package_records_dataset_and_detects_weight_changes(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "source.onnx"
            source.write_bytes(b"deterministic-test-model")
            dataset_manifest = root / "dataset_manifest.json"
            dataset_manifest.write_text(
                json.dumps(
                    {
                        "dataset_id": "dataset-test",
                        "data_sha256": "a" * 64,
                    }
                )
            )
            history = root / "history.json"
            metrics = root / "metrics.json"
            history.write_text("[]")
            metrics.write_text('{"test":{"macro_f1":0.5}}')
            package = root / "package"
            create_model_package(
                package,
                onnx_path=source,
                dataset_manifest_path=dataset_manifest,
                training_history_path=history,
                offline_metrics_path=metrics,
                model_id="model-test",
            )
            self.assertEqual(
                validate_model_package(package)["dataset_id"], "dataset-test"
            )
            (package / "model.onnx").write_bytes(b"changed")
            with self.assertRaisesRegex(ValueError, "hash mismatch"):
                validate_model_package(package)


class _FakeSource:
    def __init__(self, frame):
        self.frame = frame

    async def start(self):
        return None

    async def stop(self):
        return None

    async def wait_ready(self, timeout_s):
        return self.frame

    def latest(self):
        return self.frame

    def health(self, now_s=None):
        return SensorHealth("fake", True, frequency_hz=10)


class FaultInjectionTests(unittest.TestCase):
    def test_forced_outage_is_reported_unhealthy(self):
        frame = LaserScanFrame(
            timestamp_s=1,
            received_monotonic_s=time.monotonic(),
            frame_id="lidar",
            angle_min_rad=-1,
            angle_max_rad=1,
            angle_step_rad=1,
            range_min_m=0.1,
            range_max_m=10,
            ranges_m=(1.0, 2.0, 3.0),
            source="fake",
            sequence=1,
        )
        source = FaultInjectedLidarSource(
            _FakeSource(frame),
            {
                "seed": 1,
                "sensor_outage_probability": 1.0,
                "lidar_noise_stddev_m": 0,
                "lidar_dropout_probability": 0,
            },
        )
        self.assertIsNone(source.latest())
        self.assertFalse(source.health().healthy)
        self.assertIn("outage", source.health().message)

    def test_short_outage_retains_last_frame_within_stale_tolerance(self):
        received = time.monotonic()
        first = LaserScanFrame(
            timestamp_s=1, received_monotonic_s=received, frame_id="lidar",
            angle_min_rad=-1, angle_max_rad=1, angle_step_rad=1,
            range_min_m=0.1, range_max_m=10, ranges_m=(1.0, 2.0, 3.0),
            source="fake", sequence=1,
        )
        underlying = _FakeSource(first)
        source = FaultInjectedLidarSource(underlying, {
            "seed": 1, "sensor_outage_probability": 1.0,
            "sensor_outage_duration_s": 0.3,
            "lidar_noise_stddev_m": 0, "lidar_dropout_probability": 0,
        })
        source._cached_sequence = 1
        source._cached_frame = first
        underlying.frame = LaserScanFrame(**{**first.__dict__, "sequence": 2})
        self.assertEqual(source.latest().sequence, 1)
        self.assertTrue(source.health(now_s=received + 0.1).healthy)
        self.assertIn("stale tolerance", source.health(now_s=received + 0.1).message)

    def test_prolonged_outage_becomes_unhealthy_after_stale_budget(self):
        received = time.monotonic()
        frame = LaserScanFrame(
            timestamp_s=1, received_monotonic_s=received, frame_id="lidar",
            angle_min_rad=-1, angle_max_rad=1, angle_step_rad=1,
            range_min_m=0.1, range_max_m=10, ranges_m=(1.0,),
            source="fake", sequence=1,
        )
        underlying = _FakeSource(frame)
        source = FaultInjectedLidarSource(underlying, {
            "seed": 1, "sensor_outage_probability": 1.0,
            "sensor_outage_duration_s": 1.0,
            "lidar_noise_stddev_m": 0, "lidar_dropout_probability": 0,
        })
        source._cached_sequence = 1
        source._cached_frame = frame
        underlying.frame = LaserScanFrame(**{**frame.__dict__, "sequence": 2})
        source.latest()
        self.assertFalse(source.health(now_s=received + 0.6).healthy)

    def test_first_frame_after_outage_is_always_a_recovery_frame(self):
        frame = LaserScanFrame(
            timestamp_s=1, received_monotonic_s=time.monotonic(), frame_id="lidar",
            angle_min_rad=-1, angle_max_rad=1, angle_step_rad=1,
            range_min_m=0.1, range_max_m=10, ranges_m=(1.0,),
            source="fake", sequence=1,
        )
        underlying = _FakeSource(frame)
        source = FaultInjectedLidarSource(underlying, {
            "seed": 1, "sensor_outage_probability": 1.0,
            "sensor_outage_duration_s": 0.1,
            "lidar_noise_stddev_m": 0, "lidar_dropout_probability": 0,
        })
        source._cached_sequence = 1
        source._cached_frame = frame
        source._outage = True
        source._outage_until_s = 0.0
        underlying.frame = LaserScanFrame(**{**frame.__dict__, "sequence": 2})
        self.assertEqual(source.latest().sequence, 2)
        self.assertFalse(source._outage)


class RegistryAndComparisonTests(unittest.TestCase):
    def test_registry_scheduling_is_idempotent_and_resume_only_resets_failures(self):
        with tempfile.TemporaryDirectory() as directory:
            registry = ResearchRegistry(Path(directory) / "registry.sqlite")
            study = registry.create_study("candidate", "model-v2", "model-v1")
            matrix = tier_matrix("closed-loop")
            first = registry.ensure_runs(study, "closed-loop", matrix)
            second = registry.ensure_runs(study, "closed-loop", matrix)
            self.assertEqual(len(first), 15)
            self.assertEqual(
                [row["run_id"] for row in first], [row["run_id"] for row in second]
            )
            registry.record_metrics(first[0]["run_id"], {"collision_count": 0})
            registry.set_run_status(first[1]["run_id"], "failed", failure_reason="stop")
            registry.reset_incomplete(study, "closed-loop")
            statuses = {
                row["run_id"]: row["status"]
                for row in registry.runs(study, tier="closed-loop")
            }
            self.assertEqual(statuses[first[0]["run_id"]], "completed")
            self.assertEqual(statuses[first[1]["run_id"]], "pending")

    def test_study_queue_materializes_scenarios_and_skips_missing_champion(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            registry = ResearchRegistry(root / "registry.sqlite")
            model = root / "model"
            model.mkdir()
            registry.register_model(
                {
                    "model_id": "model-v1",
                    "onnx_sha256": "c" * 64,
                    "dataset_id": "dataset-v1",
                    "parent_model": None,
                },
                model,
            )
            study = registry.create_study("first", "model-v1")
            queue = write_run_queue(registry, study, "replay", root / "results")
            payload = json.loads(queue.read_text())
            self.assertEqual(len(payload["runs"]), 5)
            self.assertEqual(
                {row["condition"] for row in payload["runs"]}, {"candidate"}
            )
            for row in payload["runs"]:
                self.assertTrue(Path(row["scenario_manifest"]).is_file())
                self.assertIn(f"{study}/replay/results", row["result_path"])

    def test_challenge_queue_materializes_one_route_blocker_for_three_runs(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            registry = ResearchRegistry(root / "registry.sqlite")
            model = root / "model"
            model.mkdir()
            registry.register_model({
                "model_id": "model-v1",
                "onnx_sha256": "c" * 64,
                "dataset_id": "dataset-v1",
                "parent_model": None,
            }, model)
            study = registry.create_study("challenge", "model-v1")
            queue = write_run_queue(
                registry, study, "challenge", root / "results"
            )
            payload = json.loads(queue.read_text())
            self.assertEqual(len(payload["runs"]), 3)
            manifests = {row["scenario_manifest"] for row in payload["runs"]}
            self.assertEqual(len(manifests), 1)
            scenario = json.loads(Path(manifests.pop()).read_text())
            self.assertEqual(scenario["unknown_obstacles"], [])
            self.assertEqual(len(scenario["unmapped_obstacles"]), 1)
            self.assertEqual(
                scenario["unmapped_obstacles"][0]["profile"],
                "unmapped_route_blocker_v1",
            )
            self.assertEqual(
                scenario["sensor_faults"]["lidar_noise_stddev_m"], 0.0
            )
            self.assertEqual(
                scenario["sensor_faults"]["sensor_outage_probability"], 0.0
            )
            self.assertTrue(all(
                row["required_capabilities"] for row in payload["runs"]
            ))

    def test_paired_bootstrap_and_promotion_gate(self):
        runs = []
        for index in range(5):
            runs.extend(
                [
                    {
                        "scenario_id": str(index),
                        "condition": "geometric_lidar",
                        "status": "completed",
                        "metrics": {
                            "risk_f1": 0.7,
                            "traversability_iou": 0.6,
                            "inference_p95_ms": 20,
                            "collision_count": 0,
                            "safety_failure_count": 0,
                        },
                    },
                    {
                        "scenario_id": str(index),
                        "condition": "ml_lidar",
                        "status": "completed",
                        "metrics": {
                            "risk_f1": 0.8,
                            "traversability_iou": 0.65,
                            "inference_p95_ms": 18,
                            "collision_count": 0,
                            "safety_failure_count": 0,
                        },
                    },
                ]
            )
        differences = paired_differences(
            runs, "risk_f1", "ml_lidar", "geometric_lidar"
        )
        interval = bootstrap_interval(differences, samples=100)
        self.assertAlmostEqual(interval["mean"], 0.1)
        self.assertTrue(promotion_gate(runs)["passed"])
        report = study_gate_report(
            {"replay": [], "closed-loop": runs, "formal": runs}
        )
        self.assertFalse(report["passed"])
        self.assertIn("replay", report)

    def test_formal_report_preserves_fixed_multi_condition_comparisons(self):
        runs = []
        for scenario in ("a", "b"):
            for condition, f1 in (
                ("map_oracle", 1.0),
                ("geometric_lidar", 0.7),
                ("ml_lidar", 0.8),
                ("geometric_ml_fusion", 0.85),
            ):
                runs.append({
                    "scenario_id": scenario,
                    "condition": condition,
                    "status": "completed",
                    "metrics": {"risk_f1": f1, "collision_count": 0},
                })
        report = formal_comparison_report(runs, samples=100)
        self.assertEqual(report["bootstrap"]["seed"], 17)
        self.assertEqual(len(report["paired_comparisons"]), 4)
        effect = report["paired_comparisons"][
            "ml_lidar__vs__geometric_lidar"
        ]["metrics"]["risk_f1"]
        self.assertEqual(effect["count"], 2)
        self.assertAlmostEqual(effect["mean"], 0.1)
        self.assertEqual(report["descriptive"]["map_oracle"]["completed"], 2)


if __name__ == "__main__":
    unittest.main()

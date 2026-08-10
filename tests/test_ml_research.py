import json
from pathlib import Path
import tempfile
import unittest

from src.flight.safety_supervisor import SafetySupervisor
from src.ml.dataset import ResearchSample, load_dataset, validate_split_isolation
from src.ml.domain_randomization import load_ranges, sample_manifest
from src.ml.metrics import binary_iou, classification_report, latency_summary
from src.ml.protocol import experiment_matrix, load_protocol
from src.ml.research_recorder import ResearchDatasetWriter
from src.vision.training.labels import BoundingBoxLabel
from src.sensors.types import RiskEstimate


ROOT = Path(__file__).resolve().parents[1]


def sample(*, scenario="simple-1", split="train", map_id="simple", seed=1):
    return ResearchSample(
        scenario_id=scenario,
        split=split,
        map_id=map_id,
        seed=seed,
        timestamp_s=1.0,
        ranges_m=(1.0, 2.0, 3.0),
        range_max_m=10.0,
        velocity_ned_m_s=(0.1, 0.0, 0.0),
        pose_ned_m=(0.0, 0.0, -2.0),
        yaw_deg=0.0,
        risk_label="warning",
        traversability=(0.2, 0.7, 1.0),
        equipment_labels=("transformer",),
        safety_label="warning",
    )


class DatasetTests(unittest.TestCase):
    def test_writer_and_loader_preserve_versioned_sample(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "samples.jsonl"
            with ResearchDatasetWriter(path) as writer:
                writer.append(sample())
            loaded = load_dataset(path)
        self.assertEqual(loaded, [sample()])

    def test_scenario_and_map_seed_leakage_are_rejected(self):
        values = [
            sample(scenario="shared", split="train", seed=3),
            sample(scenario="shared", split="test", seed=4),
        ]
        with self.assertRaisesRegex(ValueError, "split leakage"):
            validate_split_isolation(values)
        values = [
            sample(scenario="a", split="train", seed=3),
            sample(scenario="b", split="test", seed=3),
        ]
        with self.assertRaisesRegex(ValueError, "map/seed leakage"):
            validate_split_isolation(values)


class ResearchAlgorithmTests(unittest.TestCase):
    def test_metrics_cover_classification_iou_and_latency(self):
        report = classification_report(
            ["clear", "warning"], ["clear", "danger"], ["clear", "warning", "danger"]
        )
        self.assertEqual(report["accuracy"], 0.5)
        self.assertAlmostEqual(binary_iou([1, 0, 1], [1, 1, 1]), 2 / 3)
        self.assertEqual(latency_summary([1, 2, 10])["p50_ms"], 2)

    def test_yolo_label_conversion_uses_locked_class_order(self):
        label = BoundingBoxLabel("switchgear", 10, 20, 30, 60, 100, 100)
        self.assertEqual(label.to_yolo(), "1 0.200000 0.400000 0.200000 0.400000")


class ProtocolAndSafetyTests(unittest.TestCase):
    def test_research_protocol_has_30_independent_seeds(self):
        protocol = load_protocol(ROOT / "config/perception/research_protocol.json")
        matrix = experiment_matrix(protocol)
        self.assertEqual({row["seed"] for row in matrix}, set(range(1001, 1031)))
        self.assertEqual(len(matrix), 30 * 4)

    def test_domain_randomization_is_seeded(self):
        config = load_ranges(ROOT / "config/perception/domain_randomization.json")
        first = sample_manifest(config, map_id="complex", seed=12)
        second = sample_manifest(config, map_id="complex", seed=12)
        self.assertEqual(first, second)

    def test_unhealthy_sensor_overrides_model_risk(self):
        supervisor = SafetySupervisor(stale_after_s=0.5)
        decision = supervisor.evaluate(
            {
                "sensor_healthy": False,
                "sensor_message": "stream stopped",
                "risk_level": "clear",
            }
        )
        self.assertEqual(decision.action, "hover_then_land")
        self.assertEqual(decision.speed_scale, 0.0)

    def test_low_confidence_model_is_not_accepted_as_clear(self):
        supervisor = SafetySupervisor(minimum_confidence=0.7)
        risk = RiskEstimate("clear", 0.3, None, None, 0, 2, "model", "")
        decision = supervisor.evaluate(
            {"sensor_healthy": True, "risk_level": "clear", "risk_estimate": risk}
        )
        self.assertEqual(decision.action, "hover")

    def test_degraded_sensor_hovers_until_fresh_data_recovers(self):
        supervisor = SafetySupervisor(stale_after_s=2.0)
        decision = supervisor.evaluate({
            "sensor_healthy": True,
            "sensor_message": "injected sensor stream outage within stale tolerance",
            "sensor_frame_age_s": 0.8,
            "risk_level": "clear",
        })
        self.assertEqual(decision.action, "hover")
        self.assertEqual(decision.speed_scale, 0.0)


if __name__ == "__main__":
    unittest.main()

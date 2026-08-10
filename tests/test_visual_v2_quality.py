from types import SimpleNamespace
import unittest

from src.vision.collection.quality import (
    summarize_background_route,
    summarize_target_route,
)


def _annotation(status, phase, objects=()):
    return SimpleNamespace(
        annotation_status=status,
        mission_phase=phase,
        objects=objects,
        frame_id=f"{phase}-{status}",
        image_width=100,
        image_height=100,
    )


class VisualV2QualityTests(unittest.TestCase):
    def test_background_gate_excludes_preflight_other_frames(self):
        summary = summarize_background_route(
            (
                _annotation("labelled", "other"),
                _annotation("verified_no_target", "cruise_distant"),
            )
        )
        self.assertEqual(summary["unexpected_labelled_frame_count"], 0)
        self.assertEqual(summary["verified_no_target_frame_count"], 1)

    def test_target_gate_counts_only_dataset_phases(self):
        target = SimpleNamespace(
            class_name="reactor",
            bbox_xyxy=(0, 0, 50, 50),
            truncation_status="not_truncated",
        )
        other = _annotation("labelled", "other", (target,))
        cruise = _annotation("labelled", "cruise_distant", (target,))
        frames = {
            other.frame_id: SimpleNamespace(capture_timestamp=1.0),
            cruise.frame_id: SimpleNamespace(capture_timestamp=2.0),
        }
        summary = summarize_target_route(
            (other, cruise),
            frames,
            "reactor",
        )
        self.assertEqual(summary["target_frame_count"], 1)


if __name__ == "__main__":
    unittest.main()

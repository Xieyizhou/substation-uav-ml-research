import unittest

from src.planner.active_inspection import (
    ActiveInspectionPlanner,
    ActiveInspectionPolicy,
    RuntimeMap,
)
from tests.test_active_inspection import MAP, POLICY, event


class LiveInspectionSchedulerTests(unittest.TestCase):
    def test_fixed_serpentine_ignores_equipment_utility(self):
        planner = ActiveInspectionPlanner(
            RuntimeMap.from_mapping(MAP),
            ActiveInspectionPolicy.from_mapping(POLICY),
            scheduler="fixed_serpentine",
        )
        planner.process(event())
        self.assertEqual(planner.decisions[-1]["scheduler"], "fixed_serpentine")
        self.assertEqual(planner.decisions[-1]["kind"], "exploration")

    def test_nearest_target_prefers_reachable_equipment(self):
        planner = ActiveInspectionPlanner(
            RuntimeMap.from_mapping(MAP),
            ActiveInspectionPolicy.from_mapping(POLICY),
            scheduler="nearest_target_first",
        )
        planner.process(event())
        self.assertEqual(planner.decisions[-1]["scheduler"], "nearest_target_first")
        self.assertEqual(planner.decisions[-1]["kind"], "equipment_observation")

    def test_unknown_scheduler_is_rejected(self):
        with self.assertRaises(ValueError):
            ActiveInspectionPlanner(
                RuntimeMap.from_mapping(MAP),
                ActiveInspectionPolicy.from_mapping(POLICY),
                scheduler="unknown",
            )


if __name__ == "__main__":
    unittest.main()

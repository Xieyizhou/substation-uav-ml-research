import unittest
from src.planner.active_inspection import RuntimeMap
from src.planner.executed_coverage import ExecutedCoverageTracker
from src.vision.localization.rgbd_localization import localize_bbox

class LocalizationTests(unittest.TestCase):
    def test_center_yaw_zero_projects_north(self):
        result=localize_bbox(5,[45,45,55,55],{"fx":100,"fy":100,"cx":50,"cy":50},{"east_m":2,"north_m":3,"altitude_m":4,"roll_deg":0,"pitch_deg":0,"yaw_deg":0})
        self.assertAlmostEqual(result["north_m"],8); self.assertAlmostEqual(result["east_m"],2)
    def test_yaw_ninety_projects_east(self):
        result=localize_bbox(5,[45,45,55,55],{"fx":100,"fy":100,"cx":50,"cy":50},{"east_m":2,"north_m":3,"altitude_m":4,"roll_deg":0,"pitch_deg":0,"yaw_deg":90})
        self.assertAlmostEqual(result["east_m"],7)
    def test_missing_attitude_rejected(self): self.assertIsNone(localize_bbox(5,[0,0,1,1],{"fx":1,"fy":1},{"east_m":0}))
    def test_camera_translation_is_applied_before_ned_rotation(self):
        result=localize_bbox(5,[45,45,55,55],{"fx":100,"fy":100,"cx":50,"cy":50},{"east_m":0,"north_m":0,"altitude_m":2,"roll_deg":0,"pitch_deg":0,"yaw_deg":0},{"forward_m":.18,"down_m":-.12})
        self.assertAlmostEqual(result["north_m"],5.18); self.assertAlmostEqual(result["altitude_m"],2.12)

class CoverageTests(unittest.TestCase):
    def test_only_executed_footprint_counts(self):
        grid=RuntimeMap.from_mapping({"map_name":"m","width":5,"height":5,"resolution_m":1,"obstacles":[{"x_min":2,"x_max":2,"y_min":2,"y_max":2}]})
        tracker=ExecutedCoverageTracker(grid,radius_m=1); tracker.observe(0,0)
        self.assertGreater(tracker.coverage,0); self.assertLess(tracker.coverage,1); self.assertNotIn((2,2),tracker.covered)
    def test_receipt_deterministic(self):
        grid=RuntimeMap.from_mapping({"map_name":"m","width":3,"height":3,"resolution_m":1,"obstacles":[]}); a,b=ExecutedCoverageTracker(grid),ExecutedCoverageTracker(grid); a.observe(1,1); b.observe(1,1); self.assertEqual(a.receipt(),b.receipt())

if __name__ == "__main__": unittest.main()

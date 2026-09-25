import unittest

from scripts.vision.analyze_recovery_paired_calibration import indexed, iou


def test_iou_handles_occlusion_and_disjoint_boxes():
    assert iou([0, 0, 10, 10], [0, 0, 10, 10]) == 1
    assert iou([0, 0, 4, 10], [0, 0, 10, 10]) == .4
    assert iou([0, 0, 1, 1], [2, 2, 3, 3]) == 0
    assert iou([0, 0, 0, 0], [0, 0, 0, 0]) == 0


def test_instance_matching_ignores_message_prefix_and_retains_ambiguity():
    rows = [{'annotation_id': 'message-a-instance-7-box-1'},
            {'annotation_id': 'message-b-instance-7-box-2'},
            {'annotation_id': 'message-a-instance-8-box-1'}]
    groups = indexed(rows)
    assert groups[7] == rows[:2]
    assert groups[8] == rows[2:]


class PairedCalibrationTests(unittest.TestCase):
    def test_geometry(self):
        test_iou_handles_occlusion_and_disjoint_boxes()

    def test_instance_identity(self):
        test_instance_matching_ignores_message_prefix_and_retains_ambiguity()

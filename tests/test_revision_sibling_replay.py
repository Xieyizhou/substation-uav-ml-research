import unittest
from scripts.vision.summarize_revision_siblings import inspect
from pathlib import Path


class SiblingGateTests(unittest.TestCase):
    def test_incomplete_rejected(self):
        with self.assertRaises(ValueError):inspect({'status':'semantic_blocked'},Path('/unused'))

    def test_unstable_count_rejected(self):
        with self.assertRaises(ValueError):inspect(dict(status='existing_pose_technical_checks_passed',process_cleanup_complete=True,records=[]),Path('/unused'))

    def test_alignment_and_membership_rejected_before_mask(self):
        row=dict(rgb_exact=True,skew_ms=1,lost_labels=[],added_labels=['128'],historical_deltas={'1':0})
        for change in (dict(rgb_exact=False),dict(skew_ms=34),dict(lost_labels=['1']),dict(added_labels=[]),dict(historical_deltas={'1':2})):
            with self.assertRaises(ValueError):inspect(dict(status='existing_pose_technical_checks_passed',process_cleanup_complete=True,records=[dict(row,**change)]*3),Path('/unused'))

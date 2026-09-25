import copy
import unittest
from scripts.vision.small_scale_review_gate import validate_decisions


class ReviewGate(unittest.TestCase):
    def setUp(self):
        self.frames = {'S': dict(identity='e', image_sha256='i', events=[
            dict(event_id='S:other', object_id='other', crop_sha256='c')])}
        self.decisions = [dict(event_id='S:other', object_id='other', crop_sha256='c',
            evidence_identity='e', image_sha256='i', reason='Own crop inspected',
            review_time='2026-09-14T00:00:00Z', review_nature='AI辅助审核',
            status='content_sufficient_for_bounded_research', training_admitted=False, promotable=False)]

    def test_valid(self):
        self.assertEqual(validate_decisions(self.frames, self.decisions, ['S']), 1)

    def test_missing_duplicate(self):
        for ds in ([], self.decisions * 2):
            with self.assertRaises(ValueError): validate_decisions(self.frames, ds, ['S'])

    def test_stale_and_planned_identity_substitution(self):
        for field in ('evidence_identity', 'image_sha256', 'crop_sha256', 'object_id'):
            ds = copy.deepcopy(self.decisions); ds[0][field] = 'wrong'
            with self.assertRaises(ValueError): validate_decisions(self.frames, ds, ['S'])

    def test_unknown(self):
        self.decisions[0]['status'] = 'unknown'
        with self.assertRaises(ValueError): validate_decisions(self.frames, self.decisions, ['S'])

    def test_full_frame_missing_or_duplicate(self):
        for viewed in ([], ['S', 'S']):
            with self.assertRaises(ValueError): validate_decisions(self.frames, self.decisions, viewed)


if __name__ == '__main__': unittest.main()

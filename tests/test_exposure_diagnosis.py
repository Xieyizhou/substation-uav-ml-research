import unittest
from collections import Counter
import tempfile
from pathlib import Path

from scripts.vision.exposure_protocol import QUOTAS, make_schedules
from scripts.vision.exposure_protocol import save, verify, file_sha256
from scripts.vision.exposure_metrics import missed_reason, score
from scripts.vision.finalize_exposure_diagnosis import verify_review
from scripts.vision.finalize_exposure_diagnosis import policy_checks
from scripts.vision.run_exposure_diagnosis import checked_cell
from scripts.vision.exposure_protocol import NAMES
from scripts.vision.build_exposure_review import frame_key, prediction_id


class ExposureSchedulesTests(unittest.TestCase):
    def test_same_pose_different_light_has_distinct_frame_and_prediction_identity(self):
        normal = {'view_id': 'shared_pose', 'variant': 'light_normal'}
        cool = {'view_id': 'shared_pose', 'variant': 'light_cool_low'}
        self.assertNotEqual(frame_key(normal), frame_key(cool))
        self.assertNotEqual(prediction_id(17, normal, 0), prediction_id(17, cool, 0))
        self.assertNotEqual(prediction_id(17, normal, 0), prediction_id(17, normal, 1))

    def test_exact_quotas_prefix_and_only_36_swaps(self):
        rows = [{'member_id': f'{s}:{i}', 'subset': s} for s, n in
                {'base': 66, 'regular': 48, 'bridge_positive': 48, 'hard_negative': 24}.items() for i in range(n)]
        lookup = {r['member_id']: r['subset'] for r in rows}
        for seed in (7, 17, 27):
            schedules = make_schedules(rows, seed)
            self.assertEqual(schedules, make_schedules(rows, seed))
            for arm in 'RXY':
                self.assertEqual(len(schedules[arm]), 1800)
                for stop in (600, 1800):
                    counts = Counter(schedules[arm][:stop])
                    for subset in QUOTAS[arm]:
                        values = [counts[r['member_id']] for r in rows if r['subset'] == subset]
                        self.assertLessEqual(max(values) - min(values), 1)
                for start in (0, 600, 1200):
                    chunk = schedules[arm][start:start + 600]
                    self.assertEqual(Counter(lookup[m] for m in chunk), QUOTAS[arm])
                    self.assertEqual(len(set(chunk)), 114 if arm == 'R' else 186)
            for start in (0, 600, 1200):
                differences = [(x, y) for x, y in zip(schedules['X'][start:start+600], schedules['Y'][start:start+600]) if x != y]
                self.assertEqual(len(differences), 36)
                self.assertTrue(all(lookup[x] == 'bridge_positive' and lookup[y] == 'hard_negative' for x, y in differences))

    def test_miss_priority_and_low_confidence_boundaries(self):
        truth = {'class_name': 'transformer', 'bbox_xyxy': [0, 0, 10, 10]}
        low = {**truth, 'confidence': .2}
        wrong = {**truth, 'class_name': 'reactor', 'confidence': .9}
        shifted = {**truth, 'bbox_xyxy': [5, 0, 15, 10], 'confidence': .8}
        self.assertEqual(missed_reason(truth, [low, wrong, shifted]), 'low_confidence_same_class')
        self.assertEqual(missed_reason(truth, [wrong, shifted]), 'wrong_class')
        self.assertEqual(missed_reason(truth, [shifted]), 'localization')
        self.assertEqual(missed_reason(truth, []), 'no_qualifying_retained_prediction')

    def test_duplicate_prediction_and_target_assignment_conflict(self):
        truth = [{'class_name': 'transformer', 'bbox_xyxy': b} for b in ([0,0,10,10], [1,0,11,10])]
        row = {'expected_category': 'transformer', 'target_bbox_xyxy': [1,0,11,10],
               'expected_object_id': 'second', 'view_id': 'view', 'pair_id': 'pair', 'variant': 'original', 'image_sha256': 'hash'}
        # Separate target boxes by more than the one-pixel lookup tolerance.
        truth[1]['bbox_xyxy'] = row['target_bbox_xyxy'] = [2,0,12,10]
        pred = {**truth[0], 'confidence': .8}
        scored = score(row, truth, [pred], [pred])
        self.assertTrue(scored['matching_conflict'])
        self.assertEqual(len(scored['matches']), 1)
        only = [truth[1]]
        duplicate = {**only[0], 'confidence': .8}
        scored = score(row, only, [duplicate, duplicate], [duplicate])
        self.assertEqual(scored['unmatched_prediction_count'], 1)
        self.assertFalse(scored['matching_conflict'])

    def test_changed_bytes_and_missing_review_fail_closed(self):
        with tempfile.TemporaryDirectory() as tmp:
            p = Path(tmp)/'data'
            p.write_text('frozen')
            record = save(Path(tmp)/'receipt.json', {'inputs': {str(p): file_sha256(p)}})
            verify(record)
            p.write_text('changed')
            with self.assertRaisesRegex(ValueError, 'Input bytes'):
                verify(record)
            manifest = save(Path(tmp)/'manifest.json', {'frames': [{'predictions': [{'prediction_id': 'one'}]}]})
            review = save(Path(tmp)/'review.json', {'decisions': []})
            with self.assertRaisesRegex(ValueError, 'missing'):
                verify_review(review, manifest)

    def test_retention_checks_both_references_and_each_class(self):
        import copy
        reference = {v: {'instance_recall': {'mean': .8}, 'per_class':
                     {name: {'instance_recall': {'mean': .8}} for name in NAMES}}
                     for v in ('original', 'lighting')}
        policy = {'acceptance_policy': [], 'retention': {'variants': ['original', 'lighting'], 'tolerance': .05}}
        candidate = copy.deepcopy(reference)
        candidate['lighting']['per_class']['reactor']['instance_recall']['mean'] = .75
        self.assertTrue(policy_checks(candidate, reference, reference, policy)['passed'])
        candidate['lighting']['per_class']['reactor']['instance_recall']['mean'] = .74
        result = policy_checks(candidate, reference, reference, policy)
        self.assertFalse(result['passed'])
        self.assertEqual(sum(not c['passed'] for c in result['checks']), 2)

    def test_resumed_cell_must_match_actual_exposure(self):
        with tempfile.TemporaryDirectory() as tmp:
            p = Path(tmp)
            exposure_path = p/'exposure.json'
            save(exposure_path, {'draws': ['a']*6, 'summary': {'draws': 6}})
            completion = save(p/'completion.json', {'status': 'complete', 'protocol_identity': 'frozen',
                'cell': 'R-100-7', 'exposure_path': str(exposure_path), 'optimizer_steps': 1,
                'inputs': {str(exposure_path): file_sha256(exposure_path)}})
            protocol = {'identity': 'frozen', 'schedules': {'R-100-7': ['a']*6},
                        'exposures': {'R-100-7': {'draws': 6}}}
            checked_cell(p/'completion.json', protocol)
            protocol['schedules']['R-100-7'] = ['b']*6
            with self.assertRaisesRegex(ValueError, 'exposure mismatch'):
                checked_cell(p/'completion.json', protocol)


if __name__ == '__main__':
    unittest.main()

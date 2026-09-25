import unittest
from scripts.vision.run_recovery_stratified_experiment import make_schedule, exposures, expected_object


class StratifiedSamplingTests(unittest.TestCase):
    def rows(self):
        rows = []
        for cls, map_id, n in [('transformer','simple',10), ('switchgear','simple',8),
                               ('switchgear','medium',4), ('capacitor_bank','medium',6),
                               ('reactor','complex',2)]:
            for i in range(n):
                rows.append({'expected_category': cls, 'map_id': map_id,
                             'scale': 'small' if i % 2 else 'large', 'source': 'fixture',
                             'view_id': str(len(rows)), 'truth': {'objects': [{'class_name': cls}]}})
        return rows

    def test_uniform_uses_each_image_once_per_epoch(self):
        rows = self.rows()
        for epoch in make_schedule(rows, 'uniform', 7):
            self.assertEqual(sorted(epoch), list(range(30)))

    def test_balancing_changes_exposure_without_fabricating_images(self):
        rows = self.rows()
        schedule = make_schedule(rows, 'stratified', 7)
        summary = exposures(rows, schedule)
        self.assertEqual(summary['draws'], 300)
        self.assertEqual(set(summary['by_expected_class'].values()), {75})
        self.assertEqual(summary['unique_frames'], 30)
        self.assertTrue(all(len(epoch) == 30 for epoch in schedule))
        self.assertEqual(schedule, make_schedule(rows, 'stratified', 7))
        self.assertNotEqual(schedule, make_schedule(rows, 'stratified', 17))
        switch_maps = [count for key, count in summary['by_stratum'].items() if key.startswith('switchgear:')]
        self.assertLessEqual(max(switch_maps) - min(switch_maps), 1)

    def test_missing_expected_instance_cannot_be_replaced_by_same_class(self):
        row = {'expected_category': 'transformer', 'expected_object_id': 'transformer_mid',
               'view_id': 'bad', 'truth': {'objects': [{'class_name': 'transformer',
               'annotation_id': 'fake-instance-0072-box-0000', 'bbox_xyxy': [0,0,10,10]}]}}
        with self.assertRaises(ValueError):
            expected_object(row)

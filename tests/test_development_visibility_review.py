import copy
import unittest
from scripts.vision.record_development_visibility_review import validate


class ReviewTests(unittest.TestCase):
    def setUp(self):
        self.event = dict(event_id='LOSS03', image_sha256='image', page_sha256='page',
                          object_id='cabinet', runtime_label='149', truth={'box': [1, 2, 3, 4]},
                          original_pixel_evidence_certified=True, visible_pixels=1)
        self.e = {'events': [self.event]}
        self.d = [{k: self.event[k] for k in ('event_id', 'image_sha256', 'page_sha256', 'object_id', 'runtime_label', 'truth')}]
        self.d[0].update(status='visible_but_insufficient_content', reason='Observed partial surface',
                         reviewed_at='2026-09-10', review_nature='AI-assisted', training_admitted=False, promotable=False)

    def test_valid_does_not_require_area_threshold(self):
        validate(self.e, self.d)

    def test_missing_duplicate_rejected(self):
        for ds in ([], self.d * 2):
            with self.assertRaises(ValueError): validate(self.e, ds)

    def test_stale_or_wrong_instance_rejected(self):
        for key in ('image_sha256', 'page_sha256', 'object_id', 'runtime_label', 'truth'):
            ds = copy.deepcopy(self.d)
            ds[0][key] = 'changed'
            with self.assertRaises(ValueError): validate(self.e, ds)

    def test_alignment_and_empty_mask_not_certified(self):
        for key, value in (('original_pixel_evidence_certified', False), ('visible_pixels', 0)):
            e = copy.deepcopy(self.e)
            e['events'][0][key] = value
            with self.assertRaises(ValueError): validate(e, self.d)

    def test_no_admission_or_automatic_accept(self):
        for key, value in (('training_admitted', True), ('promotable', True), ('status', 'accepted'), ('reason', '')):
            ds = copy.deepcopy(self.d)
            ds[0][key] = value
            with self.assertRaises(ValueError): validate(self.e, ds)

    def test_duplicate_evidence(self):
        with self.assertRaises(ValueError): validate({'events': [self.event, self.event]}, self.d)

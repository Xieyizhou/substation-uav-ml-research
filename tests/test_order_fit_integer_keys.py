import json
import unittest
from scripts.vision.verify_order_fit_legacy_integer_keys import verify_identity, prior


class IntegerKeyTests(unittest.TestCase):
    def record(self):
        r = {'records': [{'visible_pixels_by_label': {2: 20, 11: 4}}], 'status': 'held', 'inputs': {}}
        r['identity'] = prior.verify.__globals__['object_sha256'](r)
        return json.loads(json.dumps(r))

    def test_known_integer_roundtrip(self):
        verify_identity(self.record(), 'mask_counts')

    def test_changed_content_rejected(self):
        r = self.record(); r['status'] = 'approved'
        with self.assertRaises(ValueError): verify_identity(r, 'mask_counts')

    def test_ambiguous_keys_and_unknown_schema_rejected(self):
        r = self.record(); r['records'][0]['visible_pixels_by_label']['02'] = 3
        with self.assertRaises(ValueError): verify_identity(r, 'mask_counts')
        with self.assertRaises(ValueError): verify_identity(self.record(), 'anything')


if __name__ == '__main__': unittest.main()

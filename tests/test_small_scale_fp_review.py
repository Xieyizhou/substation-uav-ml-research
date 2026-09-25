import copy
import unittest
import tempfile
from pathlib import Path
from unittest.mock import patch
from scripts.vision import record_small_scale_fp_review as importer
from scripts.vision.record_small_scale_fp_review import validate


class ReviewGate(unittest.TestCase):
    def setUp(self):
        self.e = dict(images=[dict(image_id='F01', source=dict(image_sha256='image'),
            events=[dict(event_id='event', box_index=0)])], predictions=1, unique_images=1)
        self.o = {'F01': {0: ('cabinet_like', 'Observed panel and side.')}}

    def test_explicit(self):
        self.assertEqual(validate(self.e, self.o)[0]['status'], 'visual_content_described')

    def test_missing(self):
        with self.assertRaises(ValueError): validate(self.e, {})
        with self.assertRaises(ValueError): validate(self.e, {'F01': {}})

    def test_duplicate_event(self):
        e = copy.deepcopy(self.e)
        e['images'][0]['events'].append(dict(event_id='event', box_index=1))
        with self.assertRaises(ValueError): validate(e, {'F01': {0:self.o['F01'][0],1:self.o['F01'][0]}})

    def test_unknown_not_approved(self):
        self.assertEqual(validate(self.e, {'F01': {0: ('unknown','Occluded')}})[0]['status'], 'pending')

    def test_reason_and_count(self):
        with self.assertRaises(ValueError): validate(self.e, {'F01': {0: ('cabinet_like','')}})
        self.e['predictions'] = 2
        with self.assertRaises(ValueError): validate(self.e, self.o)

    def test_stale_page_blocks_import(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp); page=root/'page.png'; page.write_bytes(b'original evidence')
            importer.prior.frozen(root/'evidence.json',dict(self.e,inputs={str(page):importer.prior.file_sha256(page)}))
            page.write_bytes(b'changed evidence')
            with patch.object(importer,'DEST',root):
                with self.assertRaisesRegex(ValueError,'Input bytes changed'): importer.run()

    def test_changed_review_identity_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            r=importer.prior.frozen(Path(tmp)/'r.json',dict(decisions=['explicit'],inputs={}))
            r['decisions']=['different']
            with self.assertRaisesRegex(ValueError,'identity changed'): importer.prior.verify(r)


if __name__ == '__main__': unittest.main()

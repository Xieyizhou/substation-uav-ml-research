from copy import deepcopy
import unittest
from tempfile import TemporaryDirectory
from pathlib import Path
from src.ml.artifacts import object_sha256
from scripts.vision.import_transfer_pilot_review import validate


def fixture(tmp_path):
    p = tmp_path/'evidence'; p.write_text('fixed')
    from scripts.vision.import_transfer_pilot_review import prior
    e = dict(event_id='G01-original_control-00', object_id='actual-instance')
    for path, digest in [('image_path','image_sha256'),('crop_path','crop_sha256'),
                         ('page_path','page_sha256'),('source_receipt','source_receipt_sha256'),
                         ('replay_receipt','replay_receipt_sha256')]:
        e[path]=str(p); e[digest]=prior.file_sha256(p)
    d = dict(event_id=e['event_id'], object_id=e['object_id'],event_sha256=object_sha256(e),
             reason='Observed cylinder and partial base', reviewed_at='2026-09-11T00:00:00Z',
             review_nature='AI-assisted', status='diagnostic_content_reviewed',
             training_admitted=False,promotable=False,pixel_visibility_certified=False)
    return {'events':[e]},d,p


class ReviewTests(unittest.TestCase):
    def setUp(self):
        self.tmp=TemporaryDirectory(); self.addCleanup(self.tmp.cleanup)
        self.e,self.d,self.p=fixture(Path(self.tmp.name))

    def test_complete_and_unknown(self):
        self.assertTrue(validate(self.e,[self.d]))
        self.d['status']='unknown'
        self.assertFalse(validate(self.e,[self.d]))

    def test_missing_duplicate(self):
        for ds in ([],[self.d,self.d]):
            with self.assertRaises(ValueError): validate(self.e,ds)
        self.e['events'].append(deepcopy(self.e['events'][0]))
        with self.assertRaises(ValueError): validate(self.e,[self.d])

    def test_stale_or_wrong_instance(self):
        bad=deepcopy(self.d); bad['object_id']='planned-but-wrong'
        with self.assertRaises(ValueError): validate(self.e,[bad])
        self.p.write_text('changed')
        with self.assertRaises(ValueError): validate(self.e,[self.d])

    def test_no_admission(self):
        self.d['training_admitted']=True
        with self.assertRaises(ValueError): validate(self.e,[self.d])

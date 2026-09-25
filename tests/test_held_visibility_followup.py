import unittest
import tempfile
from pathlib import Path
from scripts.vision.record_held_visibility_followup import validate, OBS, file_sha256

class HeldVisibilityTests(unittest.TestCase):
    def test_explicit_coverage(self):
        self.assertEqual(len(OBS),27)
        self.assertTrue(all(reason for _,reason in OBS.values()))

    def test_missing_stale_and_unverified_pass(self):
        with tempfile.TemporaryDirectory() as folder:
            p=Path(folder)/'evidence';p.write_bytes(b'original');h=file_sha256(p)
            m={'items':[dict(review_id='T049',image_path=str(p),label_path=str(p),followup_evidence_path=str(p))]}
            d=dict(review_id='T049',decision='held_pending_instance_visibility',review_nature='AI-assisted',reason='uncertain',image_sha256=h,label_sha256=h,evidence_sha256=h)
            validate(m,[d])
            for rows in ([],[d,d],[dict(d,decision='pass')]):
                with self.assertRaises(ValueError):validate(m,rows)
            p.write_bytes(b'changed')
            with self.assertRaises(ValueError):validate(m,[d])

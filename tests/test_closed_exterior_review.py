import unittest
import tempfile
from pathlib import Path
from unittest.mock import patch
from scripts.vision import record_closed_exterior_review as review
from scripts.vision import replay_closed_exterior_repaired as repair
from src.ml.artifacts import object_sha256


class ClosedExteriorReviewTests(unittest.TestCase):
    def test_missing_duplicate_stale(self):
        event={'event_id':'x','crop_sha256':'h','truth':{'class':1}}
        ev=[{'identity':'e','events':[event]}]
        d=dict(event_id='x',crop_sha256='h',truth_identity=object_sha256(event['truth']),evidence_identity='e',reason='observed')
        review.validate(ev,[d])
        for ds in ([],[d,d],[dict(d,crop_sha256='bad')],[dict(d,evidence_identity='old')]):
            with self.assertRaises(ValueError):review.validate(ev,ds)

    def test_safe_path_deterministic(self):
        self.assertEqual(repair.safe_unit('a:b:c'),repair.safe_unit('a:b:c'))
        self.assertNotEqual(repair.safe_unit('a:b:c'),repair.safe_unit('a:b:d'))
        self.assertNotIn(':',repair.safe_unit('a:b:c'))
        with self.assertRaises(ValueError):repair.check_path('/tmp/a:b/world.sdf')
        repair.check_path('/tmp/u-abc/world.sdf')

    def test_no_fourth_attempt_without_authorization(self):
        with tempfile.TemporaryDirectory() as temp, patch.object(repair,'OUT',Path(temp)), patch.object(repair,'preflight',return_value={'identity':'f'}):
            with self.assertRaisesRegex(ValueError,'not authorized'):repair.authorization()


if __name__=='__main__':unittest.main()

import unittest
from scripts.vision import finalize_source_isolated_pilot as f


class PilotReviewTests(unittest.TestCase):
    def test_missing_duplicate_and_stale_rejected(self):
        event={'event_id':'a','crop_sha256':'abc','truth':{'bbox':[1,2,3,4]}}
        ev=[{'events':[event]}]
        good={'event_id':'a','crop_sha256':'abc','truth_identity':f.object_sha256(event['truth']),
              'reason':'explicit observation','training_admitted':False}
        f.validate_decisions(ev,[good])
        for bad in ([],[good,good],[dict(good,crop_sha256='changed')],[dict(good,truth_identity='old')]):
            with self.assertRaises(ValueError):f.validate_decisions(ev,bad)

    def test_contained_components(self):
        r=f.containment(f.OUT/'plans/layout-A/original/world.sdf')
        self.assertEqual(len(r['components']),6)
        self.assertTrue(r['all_six_inside'])

    def test_all_explicit_observations(self):
        self.assertEqual(len(f.NOTES),16)
        self.assertEqual(sum(len(v) for v in f.NOTES.values()),28)


if __name__=='__main__':unittest.main()

import unittest
from scripts.vision.replay_retained_bridge_pilot import select
from scripts.vision.run_visibility_cleanup_validation import certify

class PilotTests(unittest.TestCase):
    def frames(self):
        return [dict(lineage_id=key,variant=variant,status='source_verified_replay_pending',objects=[dict(category='switchgear',boundary_contact_within_one_pixel=edge)]) for key,edge in [('a',True),('b',False),('c',True)] for variant in ('original','neutral_bridge','background_bridge')]

    def test_whole_groups_deterministic(self):
        frames=self.frames()
        self.assertEqual(select(frames),select(list(reversed(frames))))
        self.assertEqual({f['lineage_id'] for f in select(frames)},{'a','b'})
        self.assertEqual(len(select(frames)),6)

    def test_incomplete_group_rejected(self):
        frames=self.frames();del frames[1]
        with self.assertRaises(ValueError):select(frames)

    def test_nonexact_rgb_never_certified(self):
        self.assertFalse(certify(False,True,3,0,0,True))
        self.assertFalse(certify(True,True,2,0,0,True))
        self.assertFalse(certify(True,True,3,0,1.01,True))

if __name__=='__main__':unittest.main()

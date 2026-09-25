import unittest
from scripts.vision.replay_retained_bridge_remaining import remaining

class RemainingTests(unittest.TestCase):
    def source(self):
        frames=[dict(member_id=f'{i}-{v}',lineage_id=str(i),variant=v,status='source_verified_replay_pending') for i in range(13) for v in ('original','neutral_bridge','background_bridge')]
        return dict(frames=frames),dict(frames=frames[:6])

    def test_exact_complement(self):
        trace,pilot=self.source();rows=remaining(trace,pilot)
        self.assertEqual(len(rows),33)
        self.assertFalse({r['member_id'] for r in rows}&{r['member_id'] for r in pilot['frames']})

    def test_missing_variant(self):
        trace,pilot=self.source();trace['frames'][-1]['variant']='original'
        with self.assertRaises(ValueError):remaining(trace,pilot)

    def test_blocked_source(self):
        trace,pilot=self.source();trace['frames'][-1]['status']='source_blocked'
        with self.assertRaises(ValueError):remaining(trace,pilot)

if __name__=='__main__':unittest.main()

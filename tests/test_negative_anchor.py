"""Audit the actual frozen experiment schedule, not a synthetic replacement."""
import unittest
from collections import Counter
from scripts.vision.run_negative_anchor import OUT,SOURCE,read

class AnchorScheduleTests(unittest.TestCase):
    def test_resume_seed_order(self):
        from scripts.vision.resume_negative_anchor import ordered
        from scripts.vision.exposure_protocol import verify
        p=read(OUT/'protocol.json');q=ordered(p)
        self.assertEqual(list(q['schedules']),['H-100-7','H-100-17','H-100-27'])
        self.assertEqual(q['identity'],p['identity'])

    def test_exact_quota_and_shared_positives(self):
        p=read(OUT/'protocol.json');old=read(SOURCE/'protocol.json')
        rows={r['member_id']:r for r in p['pool_rows']}
        self.assertEqual(len(p['schedules']),3)
        for key,draws in p['schedules'].items():
            counts=Counter(draws)
            self.assertEqual(len(draws),600)
            negatives={m:n for m,n in counts.items() if rows[m]['subset']=='hard_negative'}
            self.assertEqual(sum(n for m,n in negatives.items() if m.startswith('coverage:')),36)
            self.assertEqual(sum(n for m,n in negatives.items() if not m.startswith('coverage:')),72)
            self.assertEqual(set(n for m,n in negatives.items() if not m.startswith('coverage:')),{3})
            new_pairs=Counter(rows[m]['lineage_id'] for m in negatives if m.startswith('coverage:'))
            self.assertEqual(len(new_pairs),18);self.assertEqual(set(new_pairs.values()),{2})
            previous=old['schedules'][key.replace('H-','O-')]
            for i,mid in enumerate(draws):
                if rows[mid]['subset']!='hard_negative':self.assertEqual(mid,previous[i])
            self.assertEqual(p['exposures'][key]['class_instance_exposure'],old['exposures'][key.replace('H-','O-')]['class_instance_exposure'])

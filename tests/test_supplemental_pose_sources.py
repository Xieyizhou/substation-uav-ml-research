"""Frozen supplemental source cardinality and explicit review identity checks."""
import unittest
from collections import Counter
from scripts.vision.verify_supplemental_full_scene_poses import OUT, DESIGN, prior


class SupplementalSources(unittest.TestCase):
    def test_frozen_selection(self):
        p = prior.read(DESIGN / 'protocol.json')
        prior.verify(p)
        rows = p['selected']
        self.assertEqual(len(rows), 8)
        self.assertEqual(len({r['view_id'] for r in rows}), 8)
        self.assertEqual(sorted(Counter(r['category'] for r in rows).values()), [2]*4)

    def test_explicit_review_coverage(self):
        r = prior.read(OUT / 'reviewed-completion.json')
        prior.verify(r)
        e = prior.read(OUT / 'review/evidence.json')
        prior.verify(e)
        self.assertEqual(len(r['sources']), 8)
        ids = [d['review_id'] for d in r['decisions']]
        self.assertEqual(len(ids), 22)
        self.assertEqual(len(set(ids)), 22)
        self.assertEqual(set(ids), {x['review_id'] for x in e['events']})
        self.assertTrue(all(d['review_type']=='AI辅助审核' and d['reason'] for d in r['decisions']))
        self.assertFalse(r['training_ready'])

    def test_each_source_has_exact_replay(self):
        c = prior.read(OUT / 'completion.json')
        prior.verify(c)
        self.assertEqual(len(c['units']), 8)
        for unit in c['units']:
            r = prior.read(unit['replay_receipt'])
            prior.verify(r)
            self.assertEqual(r['status'], 'original_pixel_evidence_certified')
            self.assertEqual(len(r['records']), 3)
            self.assertTrue(all(x['rgb_exact'] for x in r['records']))
            self.assertTrue(all(not x['missing_targets'] for x in r['full_mask_coverage']))
            self.assertTrue(r['process_cleanup_complete'])


if __name__ == '__main__':
    unittest.main()

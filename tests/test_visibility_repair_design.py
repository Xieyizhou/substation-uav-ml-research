import unittest
from collections import Counter
from scripts.vision.design_visibility_repair_contrast import schedule,verify_pair,VARIANTS

class DesignTests(unittest.TestCase):
    def fixture(self):
        rows=[dict(member_id='common',subset='base'),dict(member_id='held',subset='bridge_positive',lineage_id='held')]
        frames=[];witness=[]
        for g,n in enumerate([20,39,18,19,24]):
            witness.append(dict(lineage_id=str(g),draws_per_600_total_images=n))
            for v in VARIANTS:
                mid=f'{g}-{v}';rows.append(dict(member_id=mid,subset='bridge_positive',lineage_id=str(g)));frames.append(dict(member_id=mid,variant=v))
        old=['common']*480+['held']*120
        return old*3,rows,witness,dict(frames=frames)

    def test_exact_common_and_variant_counts(self):
        prior,rows,w,trace=self.fixture();o,m=schedule(prior,rows,w,trace,7)
        lookup={r['member_id']:r for r in rows};parsed={r['member_id']:r.get('lineage_id') for r in rows}
        allowed={f['member_id'] for f in trace['frames']};verify_pair(o,m,prior,lookup,parsed,allowed)
        variants={f['member_id']:f['variant'] for f in trace['frames']}
        for b in range(3):self.assertEqual(Counter(variants[x] for x in m[b*600:(b+1)*600] if x!='common'),Counter({v:40 for v in VARIANTS}))
        self.assertNotIn('held',m)

    def test_deterministic(self):
        args=self.fixture()
        self.assertEqual(schedule(*args,17),schedule(*args,17))
        self.assertNotEqual(schedule(*args,17),schedule(*args,27))

    def test_tampered_common_position_rejected(self):
        prior,rows,w,trace=self.fixture();o,m=schedule(prior,rows,w,trace,7);m[0]='held'
        lookup={r['member_id']:r for r in rows}
        with self.assertRaises(ValueError):verify_pair(o,m,prior,lookup,{},set())

    def test_geometry_conflict_rejected(self):
        prior,rows,w,trace=self.fixture();o,m=schedule(prior,rows,w,trace,7)
        lookup={r['member_id']:r for r in rows};parsed={r['member_id']:r['member_id'] for r in rows}
        with self.assertRaises(ValueError):verify_pair(o,m,prior,lookup,parsed,set(lookup)-{'held'})

if __name__=='__main__':unittest.main()

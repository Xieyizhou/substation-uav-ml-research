import unittest
from scripts.vision.run_stratified_negative_control import schedule,validate,QUOTAS

class StratifiedTests(unittest.TestCase):
    def fixture(self):
        rows=[dict(member_id='positive',subset='base',lineage_id='positive',class_instances={'switchgear':1})]
        for u,n in QUOTAS.items():
            for i in range(max(n,4)):
                for v in range(2):rows.append(dict(member_id=f'{u}:{i}:{v}',subset='hard_negative',lineage_id=f'{u}:{i}',coverage_unit=u,class_instances={}))
        reference=['positive']*600
        for i,r in enumerate(rows[1:37]):reference[i*6]=r['member_id']
        return rows,reference
    def test_determinism_and_shared_positions(self):
        rows,ref=self.fixture();a,p,slots=schedule(rows,ref)
        self.assertEqual((a,p,slots),schedule(rows,ref));self.assertEqual(len(slots),36)
        self.assertTrue(all(a[i]==m for i,m in enumerate(ref) if i not in slots))
    def test_reject_shared_changes_and_duplicates(self):
        rows,ref=self.fixture();a,_,slots=schedule(rows,ref)
        b=list(a);b[1]=a[slots[0]]
        with self.assertRaises(ValueError):validate(rows,ref,b)
        b=list(a);b[slots[0]]=b[slots[1]]
        with self.assertRaises(ValueError):validate(rows,ref,b)
    def test_reject_incomplete_inventory(self):
        rows,ref=self.fixture()
        with self.assertRaises(ValueError):schedule(rows[:-1],ref)

if __name__=='__main__':unittest.main()

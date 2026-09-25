import unittest
from unittest.mock import patch
from scripts.vision.audit_material_dose_feasibility import capacity

class CapacityTests(unittest.TestCase):
    def test_source_label_and_retained_original(self):
        def row(mid,variant,pair='p',cls=1):
            return dict(member_id=mid,variant=variant,pair_id=pair,full_truth=[],class_instances={'x':cls},label_path=mid)
        p=dict(pool_rows=[row('a','original'),row('b','warm'),row('c','original','other')],schedules={'T-7':['a','a','c']})
        with patch('pathlib.Path.read_bytes',return_value=b'label'):
            r=capacity(p,'T-7');self.assertEqual(r['maximum_same_source_extra'],2);self.assertEqual(r['preserve_one_original_extra'],1)
            p['pool_rows'][1]['class_instances']={'x':2}
            self.assertEqual(capacity(p,'T-7')['maximum_same_source_extra'],0)
    def test_different_complete_labels_rejected(self):
        rows=[dict(member_id=x,variant=v,pair_id='p',full_truth=[],class_instances={},label_path=x) for x,v in [('a','original'),('b','warm')]]
        with patch('pathlib.Path.read_bytes',autospec=True,side_effect=lambda p:str(p).encode()):
            self.assertEqual(capacity(dict(pool_rows=rows,schedules={'T-7':['a']}),'T-7')['maximum_same_source_extra'],0)

if __name__=='__main__':unittest.main()

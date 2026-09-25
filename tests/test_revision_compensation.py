import sys
import unittest
from pathlib import Path
from scripts.vision.revision_compensation_feasibility import solve,POLICY,read


class CountTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        p=read(POLICY/'protocol.json')
        path=next(Path(k) for k in p['inputs'] if k.endswith('scipy/__init__.py'))
        sys.path.insert(0,str(path.parent.parent))

    def setUp(self):
        self.rows=[dict(member_id='a',subset='positive',class_instances={'reactor':1}),
                   dict(member_id='b',subset='positive',class_instances={'reactor':1}),
                   dict(member_id='n',subset='hard_negative',class_instances={})]
        self.old={'a':5,'b':5,'n':2}

    def test_compensation_and_fixed_negative(self):
        r=solve(self.rows,self.old,{'a'},{'b'},{})
        self.assertEqual(r['counts'],{'a':0,'b':10,'n':2})

    def test_unreviewed_cannot_increase(self):
        self.assertEqual(solve(self.rows,self.old,{'a'},set(),{})['status'],'infeasible')

    def test_risk_group_cap(self):
        self.assertEqual(solve(self.rows,self.old,{'a'},{'b'},{'risk':['b']})['status'],'infeasible')

    def test_zero_not_revived(self):
        self.old['b']=0
        self.assertEqual(solve(self.rows,self.old,{'a'},{'b'},{})['status'],'infeasible')

import unittest
from scripts.vision.finalize_l05_hold_block import validate_counts,require_quality_ready

class HoldGateTests(unittest.TestCase):
    def setUp(self):
        self.rows=[dict(member_id=m,subset=s,class_instances=c) for m,s,c in [('a','regular',{'reactor':1}),('b','regular',{'reactor':1}),('n','hard_negative',{}),('z','regular',{'reactor':1})]]
        self.old={'a':3,'b':2,'n':2,'z':0};self.new={'a':0,'b':5,'n':2,'z':0}
    def check(self,new,groups=None):validate_counts(self.rows,self.old,new,{'a'},{'b'},groups or {})
    def test_valid(self):self.check(self.new)
    def test_held_restore(self):
        with self.assertRaises(ValueError):self.check(dict(self.new,a=1,b=4))
    def test_zero_restore(self):
        with self.assertRaises(ValueError):self.check(dict(self.new,z=1,b=4))
    def test_negative_changed(self):
        with self.assertRaises(ValueError):self.check(dict(self.new,n=1))
    def test_quota_changed(self):
        with self.assertRaises(ValueError):self.check(dict(self.new,b=4))
    def test_risk_cap(self):
        with self.assertRaises(ValueError):self.check(self.new,{'risk':['b']})
    def test_block_cannot_be_ready(self):
        with self.assertRaises(ValueError):require_quality_ready('blocked')
    def test_technical_pass_not_quality_approval(self):self.assertFalse(require_quality_ready('all_replayed_review_pending'))

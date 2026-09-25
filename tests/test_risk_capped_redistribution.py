import unittest
from scripts.vision.check_risk_capped_redistribution import validate_capped

class CapTests(unittest.TestCase):
    def setUp(self):
        self.rows=[dict(member_id=m,subset=s,class_instances=c) for m,s,c in [
            ('held','base',{'reactor':1}),('kept','base',{'reactor':1}),('risk','base',{'reactor':1}),('negative','hard_negative',{})]]
        self.old={'held':100,'kept':2000,'risk':114,'negative':486}
        self.new={'held':0,'kept':2100,'risk':114,'negative':486}
    def check(self,new,risks={'risk'}):validate_capped(self.rows,self.old,new,{'held'},risks)
    def test_equal_cap_passes(self):self.check(self.new)
    def test_reduction_passes(self):self.check({**self.new,'risk':100,'kept':2114})
    def test_balanced_risk_increase_fails(self):
        with self.assertRaises(ValueError):self.check({**self.new,'risk':115,'kept':2099})
    def test_bad_risk_identity_fails(self):
        for risks in [set(),{'missing'},{'held'}]:
            with self.assertRaises(ValueError):self.check(self.new,risks)
    def test_other_constraints_still_apply(self):
        for changes in [{'held':1,'kept':2099},{'negative':485,'kept':2101},{'risk':114.0}]:
            with self.assertRaises(ValueError):self.check({**self.new,**changes})

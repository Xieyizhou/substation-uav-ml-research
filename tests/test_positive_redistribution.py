import unittest
from scripts.vision.relax_hold_compensation import validate_counts

class RedistributionTests(unittest.TestCase):
    def setUp(self):
        self.rows=[dict(member_id=m,subset=s,class_instances=c) for m,s,c in [
            ('held','base',{'reactor':1}),('kept','base',{'reactor':1}),('negative','hard_negative',{})]]
        self.old={'held':100,'kept':2114,'negative':486}
        self.new={'held':0,'kept':2214,'negative':486}
    def test_exact_witness_passes(self):validate_counts(self.rows,self.old,self.new,{'held'})
    def test_negative_change_rejected(self):
        with self.assertRaises(ValueError):validate_counts(self.rows,self.old,{**self.new,'negative':485,'kept':2215},{'held'})
    def test_held_and_fractional_rejected(self):
        for changes in [{'held':1,'kept':2213},{'kept':2214.0}]:
            with self.assertRaises(ValueError):validate_counts(self.rows,self.old,{**self.new,**changes},{'held'})
    def test_full_class_mismatch_rejected(self):
        self.rows[0]['class_instances']['transformer']=1
        with self.assertRaises(ValueError):validate_counts(self.rows,self.old,self.new,{'held'})
    def test_missing_duplicate_rejected(self):
        with self.assertRaises(ValueError):validate_counts(self.rows,self.old,{'kept':2214,'negative':486},{'held'})
        with self.assertRaises(ValueError):validate_counts(self.rows*2,self.old,self.new,{'held'})

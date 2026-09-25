import unittest
from collections import Counter
from scripts.vision.lineage_capped_control import sequence,check_policy

class SequenceTests(unittest.TestCase):
    def setUp(self):
        self.rows=[dict(member_id=m,subset=s,class_instances={'reactor':1} if s=='base' else {}) for m,s in [('a','base'),('b','base'),('c','base'),('n','hard_negative')]]
        self.original=['a','b','c','n','b','c']*450;self.counts={'a':0,'b':1125,'c':1125,'n':450}
    def test_deterministic_counts_and_negative_positions(self):
        x=sequence(self.rows,self.original,self.counts,7)
        self.assertEqual(x,sequence(self.rows,self.original,self.counts,7));self.assertEqual(Counter(x),+Counter(self.counts))
        self.assertTrue(all(x[i]=='n' for i,m in enumerate(self.original) if m=='n'))
    def test_subset_mismatch_rejected(self):
        with self.assertRaises(ValueError):sequence(self.rows,self.original,{**self.counts,'b':1124,'n':451},7)
    def test_group_cap_rejects_member_transfer(self):
        with self.assertRaises(ValueError):check_policy(self.rows,Counter(self.original),self.counts,{'a'},{'n'},{'risk':['b','c']})
    def test_no_risk_increment(self):
        with self.assertRaises(ValueError):check_policy(self.rows,Counter(self.original),self.counts,{'a'},{'b'},{})

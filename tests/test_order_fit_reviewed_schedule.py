import unittest
from collections import Counter
from scripts.vision.order_fit_reviewed_schedule import replace


class ReviewedScheduleTests(unittest.TestCase):
    def setUp(self):
        self.rows={k:dict(subset='base',class_instances={'reactor':1}) for k in ('a','b','c')}
        self.rows['n']=dict(subset='negative',class_instances={})
        self.old=['a','n','b','b','n','a','c','c']
        self.target={'a':4,'b':0,'c':2,'n':2}

    def test_exact_counts_and_negative_positions(self):
        new=replace(self.old,self.target,self.rows,'seed-7')
        self.assertEqual(dict(Counter(new)),{k:v for k,v in self.target.items() if v})
        self.assertEqual([i for i,x in enumerate(new) if x=='n'],[1,4])
        self.assertTrue(all(new[i]==x for i,x in enumerate(self.old) if x!='b'))

    def test_reproducible(self):
        self.assertEqual(replace(self.old,self.target,self.rows,'seed-17'),replace(self.old,self.target,self.rows,'seed-17'))

    def test_cross_subset_budget_rejected(self):
        self.rows['a']['subset']='other'
        with self.assertRaises(ValueError):replace(self.old,self.target,self.rows,'seed-7')

    def test_negative_replacement_rejected(self):
        target=dict(self.target);target['n']=1;target['a']=5
        with self.assertRaises(ValueError):replace(self.old,target,self.rows,'seed-7')


if __name__=='__main__':unittest.main()

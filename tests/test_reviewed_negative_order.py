import unittest
from collections import Counter
from scripts.vision.reviewed_negative_order_control import compensated,reorder

class ReviewedOrderTests(unittest.TestCase):
    def setUp(self):
        self.held={f'held-{i}' for i in range(8)}
        self.clear=[f'clear-{i}' for i in range(4)]
        self.seq=['old']*2700+(sorted(self.held)+self.clear)*5+[f'negative-{i}' for i in range(120)]
    def test_whole_frame_holds_and_fixed_context(self):
        for seed in (7,17,27):
            new=compensated(self.seq,self.held,self.clear,seed)
            self.assertFalse(set(new)&self.held)
            self.assertEqual(new[:2700],self.seq[:2700]);self.assertEqual(new[2760:],self.seq[2760:])
            self.assertEqual([new.count(m) for m in self.clear],[15]*4)
            self.assertEqual(new,compensated(self.seq,self.held,self.clear,seed))
    def test_intact_batch_multiset_and_negative_spacing(self):
        a=compensated(self.seq,self.held,self.clear,7);b=reorder(a)
        chunks=lambda s:[tuple(s[i:i+6]) for i in range(0,len(s),6)]
        self.assertEqual(Counter(chunks(a)),Counter(chunks(b)))
        self.assertEqual([i for i,x in enumerate(chunks(b)) if x[0].startswith('negative')],list(range(23,480,24)))
    def test_incorrect_population_and_positions_rejected(self):
        for seq,clear in [(self.seq[:-1],self.clear),(self.seq,self.clear[:3]),(self.seq,[self.clear[0]]*4)]:
            with self.assertRaises(ValueError):compensated(seq,self.held,clear,7)
        bad=list(self.seq);bad[0],bad[2700]=bad[2700],bad[0]
        with self.assertRaises(ValueError):compensated(bad,self.held,self.clear,7)

if __name__=='__main__':unittest.main()

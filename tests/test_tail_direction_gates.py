import copy,unittest
from scripts.vision.record_bn_review import DEST,prior,validate
from scripts.vision.preflight_tail_interleaving import permutation

class DirectionGates(unittest.TestCase):
    def test_batch_permutation(self):
        p=permutation();self.assertEqual(sorted(p),list(range(1000)))
        self.assertEqual([x for x in p if x<900],list(range(900)))
        self.assertEqual([x for x in p if x>=900],list(range(900,1000)))
        for i in range(0,1000,50):self.assertEqual(sum(x>=900 for x in p[i:i+50]),5)
    def test_review_valid(self):
        e=prior.read(DEST/'evidence.json');r=prior.read(DEST/'review.json');validate(e,r['decisions'])
    def test_missing_duplicate_stale(self):
        e=prior.read(DEST/'evidence.json');ds=prior.read(DEST/'review.json')['decisions']
        for bad in (ds[:-1],ds[:-1]+[ds[0]]):
            with self.assertRaises(ValueError):validate(e,bad)
        bad=copy.deepcopy(ds);bad[0]['event_sha256']='stale'
        with self.assertRaises(ValueError):validate(e,bad)
    def test_unknown_not_approved(self):
        e=prior.read(DEST/'evidence.json');ds=copy.deepcopy(prior.read(DEST/'review.json')['decisions'])
        next(d for d in ds if d['content']=='unknown')['status']='observed_not_admitted'
        with self.assertRaises(ValueError):validate(e,ds)

if __name__=='__main__':unittest.main()

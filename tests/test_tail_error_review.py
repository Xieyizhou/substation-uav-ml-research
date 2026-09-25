import copy
import unittest
from scripts.vision.preflight_tail_interleaving import OUT,prior
from scripts.vision.record_bn_review import validate

# The historical evaluator is a mutable adapter shared with IL/ILR/gray runs.
# Select this experiment's immutable location directly, not its import-time
# adapter state. Do not change the archived evidence or its hashes.
DEST=OUT/'evaluation/error-review-v1'

class ReviewGate(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.e=prior.read(DEST/'evidence.json')
        cls.ds=prior.read(DEST/'review.json')['decisions']
    def test_complete(self):validate(self.e,self.ds)
    def test_missing(self):
        with self.assertRaises(ValueError):validate(self.e,self.ds[:-1])
    def test_duplicate(self):
        with self.assertRaises(ValueError):validate(self.e,self.ds[:-1]+[self.ds[0]])
    def test_stale(self):
        ds=copy.deepcopy(self.ds);ds[0]['page_sha256']='0'*64
        with self.assertRaises(ValueError):validate(self.e,ds)
    def test_changed_prediction(self):
        e=copy.deepcopy(self.e);e['events'][0]['events'][0]['prediction']['confidence']=0
        with self.assertRaises(ValueError):validate(e,self.ds)
    def test_unknown(self):
        ds=copy.deepcopy(self.ds);next(d for d in ds if d['content']=='unknown')['status']='observed_not_admitted'
        with self.assertRaises(ValueError):validate(self.e,ds)
    def test_no_visibility_certification(self):
        ds=copy.deepcopy(self.ds);ds[0]['pixel_visibility_certified']=True
        with self.assertRaises(ValueError):validate(self.e,ds)

if __name__=='__main__':unittest.main()

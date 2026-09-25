"""Exercise current explicit-review gate without altering frozen evidence."""
import copy
import unittest
from scripts.vision.record_physical_light_error_review import DEST, prior, validate

class ReviewGate(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.e = prior.read(DEST/'evidence.json')
        cls.ds = prior.read(DEST/'review.json')['decisions']

    def test_complete(self):
        validate(self.e, self.ds)

    def test_missing(self):
        with self.assertRaises(ValueError): validate(self.e, self.ds[:-1])

    def test_duplicate(self):
        with self.assertRaises(ValueError): validate(self.e, self.ds[:-1]+[self.ds[0]])

    def test_stale_evidence(self):
        ds=copy.deepcopy(self.ds); ds[0]['evidence_sha256']='0'*64
        with self.assertRaises(ValueError): validate(self.e, ds)

    def test_changed_prediction(self):
        ds=copy.deepcopy(self.ds); ds[0]['prediction']={}
        with self.assertRaises(ValueError): validate(self.e, ds)

    def test_unknown_not_approved(self):
        ds=copy.deepcopy(self.ds)
        next(d for d in ds if d['content']=='unknown_instance_content')['status']='observed_not_admitted'
        with self.assertRaises(ValueError): validate(self.e, ds)

if __name__=='__main__': unittest.main()

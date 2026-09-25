import unittest
from scripts.vision.record_retention_regression_review import validate

class ReviewGateTests(unittest.TestCase):
    def test_missing(self):
        with self.assertRaises(ValueError):validate([{'event_id':'P01'}],{})
    def test_duplicate(self):
        with self.assertRaises(ValueError):validate([{'event_id':'P01'}]*2,{'P01':'observed'})

if __name__=='__main__':unittest.main()

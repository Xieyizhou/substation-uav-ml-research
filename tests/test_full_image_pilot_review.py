import unittest
from scripts.vision.record_full_image_pilot_review import validate_observations

class ReviewTests(unittest.TestCase):
    def test_complete(self):
        validate_observations([{'frame_id':'a','objects':[{'object_id':'b'}]}],{'a':{'b':'seen'}})
    def test_missing(self):
        with self.assertRaises(ValueError):validate_observations([{'frame_id':'a','objects':[]}],{})
    def test_duplicate(self):
        f={'frame_id':'a','objects':[]}
        with self.assertRaises(ValueError):validate_observations([f,f],{'a':{}})
    def test_missing_object(self):
        with self.assertRaises(ValueError):validate_observations([{'frame_id':'a','objects':[{'object_id':'b'}]}],{'a':{}})

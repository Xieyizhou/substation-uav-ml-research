import unittest
from scripts.vision.candidate29_review_pages import raw_boxes
from scripts.vision.record_candidate29_screen import validate_screen

class CandidateAuditTests(unittest.TestCase):
    def test_raw_epsilon_preserved(self):
        msg={'annotatedBox':[{'label':1,'box':{'minCorner':{},'maxCorner':{'x':5,'y':1080.0000305175781}}}]}
        self.assertEqual(raw_boxes(msg)['1'],[0,0,5,1080.0000305175781])
    def test_duplicate_raw_label(self):
        b={'label':1,'box':{'minCorner':{},'maxCorner':{'x':5,'y':6}}}
        with self.assertRaises(ValueError):raw_boxes({'annotatedBox':[b,b]})
    def test_missing_screen(self):
        with self.assertRaises(ValueError):validate_screen([],['A01'])
    def test_duplicate_screen(self):
        d={'pair_id':'A01','reason':'inspected','training_approved':False}
        with self.assertRaises(ValueError):validate_screen([d,d],['A01'])
    def test_screen_cannot_approve(self):
        with self.assertRaises(ValueError):validate_screen([dict(pair_id='A01',reason='inspected',training_approved=True)],['A01'])
    def test_explicit_screen(self):validate_screen([dict(pair_id='A01',reason='inspected',training_approved=False)],['A01'])

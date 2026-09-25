import copy
import unittest
from scripts.vision.analyze_neutral_gray_results import group
from scripts.vision.summarize_neutral_gray_fit import bucket


class SplitAnalysis(unittest.TestCase):
    def setUp(self):
        self.row=dict(planned_truth_index=1, truth=[{'class_name':'switchgear'},{'class_name':'reactor'}],
                      matches=[{'truth_index':0,'prediction_index':0}])
    def test_own_planned_identity(self):
        self.assertEqual(group([self.row],True)['hits'],0)
        self.assertEqual(group([self.row],False)['hits'],1)
    def test_zero_denominator(self):self.assertIsNone(group([],True)['recall'])
    def test_duplicate_truth_rejected(self):
        self.row['matches'].append({'truth_index':0,'prediction_index':1})
        with self.assertRaises(ValueError):group([self.row],False)
    def test_duplicate_prediction_rejected(self):
        self.row['matches'].append({'truth_index':1,'prediction_index':0})
        with self.assertRaises(ValueError):group([self.row],False)
    def test_size_at_input_resolution(self):
        self.assertEqual(bucket({'bbox_xyxy':[0,0,95,400]},(1920,1080)),'<32')
        self.assertEqual(bucket({'bbox_xyxy':[0,0,96,400]},(1920,1080)),'32-64')
        self.assertEqual(bucket({'bbox_xyxy':[0,0,192,400]},(1920,1080)),'>=64')


if __name__=='__main__':unittest.main()

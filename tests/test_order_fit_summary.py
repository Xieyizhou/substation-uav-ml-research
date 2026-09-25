import unittest
from scripts.vision.summarize_small_scale_order_fit import stats
from scripts.vision.index_order_fit_reviews import nodes


class FitSummaryTests(unittest.TestCase):
    def test_empty_not_perfect(self):
        r=stats([])
        self.assertIsNone(r['negative_fpr']);self.assertEqual(r['recall'],{})

    def test_full_instances_and_negative_frames(self):
        positive=dict(truth=[{'class_name':'reactor'},{'class_name':'reactor'}],matches=[{'class_name':'reactor'}],
            predictions=[{}],misses=[dict(reason='wrong_class',formal_matching_competition=False)])
        negative=dict(truth=[],matches=[],predictions=[{},{}],misses=[])
        r=stats([positive,negative]);self.assertEqual(r['recall']['reactor'],.5)
        self.assertEqual(r['negative_frames_with_prediction'],1);self.assertEqual(r['negative_fpr'],1)

    def test_index_preserves_nested_location(self):
        r=list(nodes({'decisions':[{'reason':'explicit'}]}))
        self.assertEqual(r[-1][0],'$.decisions[0]')

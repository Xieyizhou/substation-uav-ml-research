import copy
import unittest

from scripts.vision.audit_canonical_recovery_increment import adapt_checked, framing_reasons
from src.vision.collection.gazebo_truth import parse_gazebo_truth_message


def fixture():
    raw={'header':{'stamp':{'sec':1},'data':[{'key':'seq','value':['1']}]},
         'annotatedBox':[{'label':69,'box':{'minCorner':{'x':10,'y':20},'maxCorner':{'x':1917,'y':100}}}]}
    truth=parse_gazebo_truth_message(raw,topic='/research_camera/boxes',width=1920,height=1080,receive_index=1)
    return {'raw_truth':raw,'truth':truth.to_record()}


class CanonicalRecoveryIncrementTests(unittest.TestCase):
    def test_explicit_conversion_preserves_source_and_checks_right_edge(self):
        view=fixture()
        original=copy.deepcopy(view)
        adapted=adapt_checked(view,{'annotation_mode':'visible_2d'})
        self.assertEqual(view,original)
        self.assertEqual(adapted['objects'][0]['bbox_xyxy'],[10,20,1918,101])
        reasons=framing_reasons(adapted,{'minimum_border_margin_px':2,
                                      'maximum_bbox_width_fraction':.9,'maximum_bbox_height_fraction':.9})
        self.assertEqual(reasons[0]['reason'],'bbox_touches_frame_boundary')

    def test_missing_source_annotation_is_not_accepted(self):
        view=fixture()
        view['truth']['objects']=[]
        with self.assertRaisesRegex(ValueError,'raw_truth_object_count_mismatch'):
            adapt_checked(view,{'annotation_mode':'visible_2d'})

    def test_changed_class_is_not_accepted(self):
        view=fixture()
        view['truth']['objects'][0]['class_name']='reactor'
        with self.assertRaisesRegex(ValueError,'raw_truth_class_mismatch'):
            adapt_checked(view,{'annotation_mode':'visible_2d'})

    def test_unspecified_annotation_mode_is_not_silently_converted(self):
        with self.assertRaisesRegex(ValueError,'explicit visible_2d'):
            adapt_checked(fixture(),{})


if __name__=='__main__':
    unittest.main()

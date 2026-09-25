import copy
import unittest
from scripts.vision.audit_brightness_lr_retention import validate_current_losses


class CurrentLosses(unittest.TestCase):
    def fixture(self):
        truth={'class_name':'reactor','annotation_id':'stable','bbox_xyxy':[1,2,30,40]}
        miss={'truth_index':0,'reason':'low_confidence'}
        row=dict(view_id='v',variant='original',pair_id='pair',image_sha256='rgb',truth=[truth],matches=[],misses=[miss])
        old={**row,'matches':[{'truth_index':0}],'misses':[]}
        events=[dict(kind='LOSS',source={'image_sha256':'rgb'},truth=truth,events=[dict(seed=s,miss=miss) for s in (7,17,27)])]
        transitions=[dict(seed=s,pair_id='pair',variant='original',truth=truth,state='loss') for s in (7,17,27)]
        return dict(events=events,all_transitions=transitions),[{'rows':[row]}]*3,[{'rows':[old]}]*3

    def test_complete(self):
        validate_current_losses(*self.fixture())

    def test_omission_or_repetition_rejected(self):
        e,new,old=self.fixture()
        for field in ('events','all_transitions'):
            for replacement in ([],e[field]*2):
                bad=copy.deepcopy(e);bad[field]=replacement
                with self.assertRaises(ValueError):validate_current_losses(bad,new,old)

    def test_truth_or_pixels_not_array_index_identity(self):
        e,new,old=self.fixture()
        for field,value in [('image_sha256','other'),('truth',[{'class_name':'reactor','annotation_id':'other'}])]:
            bad=copy.deepcopy(new);bad[0]['rows'][0][field]=value
            with self.assertRaises(ValueError):validate_current_losses(e,bad,old)


if __name__=='__main__':unittest.main()

import copy
import unittest
from unittest.mock import patch
from scripts.vision.record_clear_context_lr_review import validate_material


class MaterialReviewTests(unittest.TestCase):
    def setUp(self):
        self.t={'class_name':'switchgear','bbox_xyxy':[0,0,10,10]}
        self.e={'frames':[dict(frame_id='f',truth=[self.t],image_path='i',image_sha256='ih',evidence_path='p',evidence_sha256='ph')]}
        self.d=[dict(frame_id='f',truth_index=0,truth=self.t,reason='Observed full body',review_nature='AI辅助审核',image_sha256='ih',evidence_sha256='ph')]
    def check(self,rows,hashes=None):
        with patch('scripts.vision.record_clear_context_lr_review.file_sha256',side_effect=lambda p:(hashes or {'i':'ih','p':'ph'})[p]):validate_material(self.e,rows)
    def test_valid(self):self.check(self.d)
    def test_missing_duplicate(self):
        for d in ([],self.d*2):
            with self.assertRaises(ValueError):self.check(d)
    def test_stale_bytes(self):
        for h in ({'i':'changed','p':'ph'},{'i':'ih','p':'changed'}):
            with self.assertRaises(ValueError):self.check(self.d,h)
    def test_truth_or_reason_changed(self):
        for field,value in [('truth',{'class_name':'reactor'}),('reason',''),('review_nature','automatic pass')]:
            d=copy.deepcopy(self.d);d[0][field]=value
            with self.assertRaises(ValueError):self.check(d)


if __name__=='__main__':unittest.main()

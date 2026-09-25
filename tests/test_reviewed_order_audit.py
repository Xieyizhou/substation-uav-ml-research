import copy
import unittest
from unittest.mock import patch
from scripts.vision.audit_reviewed_order_results import validate_review

class ReviewGateTests(unittest.TestCase):
    def setUp(self):
        self.pred={'confidence':.8,'bbox_xyxy':[1,2,3,4]}
        self.e={'frames':[dict(image_path='image',image_sha256='i',evidence_path='page',evidence_sha256='e',events=[dict(event_id='fp-00',prediction=self.pred)])]}
        self.d=[dict(event_id='fp-00',prediction=self.pred,image_sha256='i',evidence_sha256='e',reason='Reviewed actual crop',review_nature='AI辅助审核')]
    def check(self,d,hashes=None):
        with patch('scripts.vision.audit_reviewed_order_results.file_sha256',side_effect=lambda p:(hashes or {'image':'i','page':'e'})[p]):validate_review(self.e,d)
    def test_valid(self):self.check(self.d)
    def test_missing_duplicate(self):
        for rows in ([],self.d*2):
            with self.assertRaises(ValueError):self.check(rows)
    def test_stale_image_or_page(self):
        for hashes in ({'image':'changed','page':'e'},{'image':'i','page':'changed'}):
            with self.assertRaises(ValueError):self.check(self.d,hashes)
    def test_changed_prediction(self):
        d=copy.deepcopy(self.d);d[0]['prediction']['confidence']=.9
        with self.assertRaises(ValueError):self.check(d)

if __name__=='__main__':unittest.main()

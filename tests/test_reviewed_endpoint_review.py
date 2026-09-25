import tempfile
import unittest
from pathlib import Path
from scripts.vision.record_reviewed_endpoint_review import validate,prior
from scripts.vision.build_reviewed_endpoint_errors import truth_key

class ReviewedEndpointTests(unittest.TestCase):
    def test_truth_is_not_array_position(self):
        a=dict(class_name='reactor',bbox_xyxy=[1,2,3,4],annotation_id='instance-a')
        b=dict(a,annotation_id='instance-b')
        self.assertNotEqual(truth_key(a),truth_key(b))
        self.assertNotEqual(truth_key(a),truth_key(dict(a,bbox_xyxy=[1,2,3,5])))
    def test_missing_and_duplicate_decisions(self):
        d=dict(review_id='M01',reason='visible cylinder',review_nature='AI辅助审核',evidence_hashes={})
        validate([d],['M01'],'review_id')
        for rows in ([],[d,d]):
            with self.assertRaises(ValueError):validate(rows,['M01'],'review_id')
    def test_stale_evidence(self):
        with tempfile.TemporaryDirectory() as tmp:
            p=Path(tmp)/'evidence';p.write_text('original')
            d=dict(review_id='M01',reason='visible cylinder',review_nature='AI辅助审核',evidence_hashes={str(p):prior.file_sha256(p)})
            validate([d],['M01'],'review_id');p.write_text('changed')
            with self.assertRaises(ValueError):validate([d],['M01'],'review_id')
    def test_reason_required(self):
        with self.assertRaises(ValueError):validate([dict(review_id='M01',reason='',review_nature='AI辅助审核',evidence_hashes={})],['M01'],'review_id')

if __name__=='__main__':unittest.main()

import copy
import tempfile
import unittest
from pathlib import Path
from scripts.vision.record_budget_error_review import validate,file_sha256

class BudgetReviewTests(unittest.TestCase):
    def test_review_identity_and_staleness(self):
        with tempfile.TemporaryDirectory() as temp:
            p=Path(temp)/'evidence';p.write_bytes(b'fixture')
            item=dict(item_id='N01',bbox_xyxy=[0,0,10,10],source=dict(image_path=str(p)),evidence_path=str(p))
            manifest=dict(items=[item]);decision=dict(item_id='N01',bbox_xyxy=item['bbox_xyxy'],reason='Explicit observation',review_nature='AI-assisted',decision='reviewed',image_sha256=file_sha256(p),evidence_sha256=file_sha256(p))
            validate(manifest,[decision])
            for rows in ([],[decision,decision]):
                with self.assertRaises(ValueError):validate(manifest,rows)
            for key,value in [('bbox_xyxy',[1,1,2,2]),('reason',''),('decision','pending'),('review_nature','automatic')]:
                altered=copy.deepcopy(decision);altered[key]=value
                with self.assertRaises(ValueError):validate(manifest,[altered])
            p.write_bytes(b'changed')
            with self.assertRaises(ValueError):validate(manifest,[decision])

if __name__=='__main__':unittest.main()

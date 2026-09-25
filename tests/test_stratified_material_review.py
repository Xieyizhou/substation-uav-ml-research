import copy
import tempfile
import unittest
from pathlib import Path
from scripts.vision.record_stratified_material_review import validate,file_sha256

class MaterialReviewTests(unittest.TestCase):
    def test_identity_and_missing_decisions(self):
        with tempfile.TemporaryDirectory() as temp:
            p=Path(temp)/'image';p.write_bytes(b'fixture');h=file_sha256(p)
            item=dict(item_id='P1',bbox_xyxy=[0,0,10,10],evidence_path=str(p),evidence_sha256=h,sources={v:dict(image_path=str(p),image_sha256=h) for v in ('material','original')})
            m=dict(items=[item]);d=dict(item_id='P1',bbox_xyxy=item['bbox_xyxy'],evidence_sha256=h,image_hashes={v:h for v in item['sources']},decision='reviewed',review_nature='AI-assisted',reason='Observed actual content')
            validate(m,[d])
            for ds in ([],[d,d]):
                with self.assertRaises(ValueError):validate(m,ds)
            bad=copy.deepcopy(d);bad['bbox_xyxy']=[1,1,2,2]
            with self.assertRaises(ValueError):validate(m,[bad])
            bad=copy.deepcopy(d);bad['reason']=''
            with self.assertRaises(ValueError):validate(m,[bad])
            p.write_bytes(b'stale')
            with self.assertRaises(ValueError):validate(m,[d])

if __name__=='__main__':unittest.main()

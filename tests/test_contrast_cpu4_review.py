import copy
import tempfile
import unittest
from pathlib import Path
from scripts.vision.record_contrast_cpu4_review import validate_positive
from src.ml.artifacts import file_sha256

class ReviewTests(unittest.TestCase):
    def test_missing_duplicate_and_stale(self):
        with tempfile.TemporaryDirectory() as d:
            p=Path(d)/'evidence';p.write_text('image')
            t={'bbox_xyxy':[1,2,3,4],'class_name':'reactor'};diag={'reason':'low_confidence_same_class'}
            frame=dict(frame_id='f',image_path=str(p),image_sha256=file_sha256(p),evidence_path=str(p),evidence_sha256=file_sha256(p),events=[dict(seed=7,truth_index=0,truth=t,diagnosis=diag)])
            e={'frames':[frame]};decision=dict(frame_id='f',seed=7,truth_index=0,truth=t,diagnosis=diag,reason='visible cylinder',review_nature='AI辅助审核',image_sha256=file_sha256(p),evidence_sha256=file_sha256(p))
            validate_positive(e,[decision])
            for ds in ([],[decision,decision]):
                with self.assertRaises(ValueError):validate_positive(e,ds)
            bad=copy.deepcopy(decision);bad['truth']['bbox_xyxy'][0]=0
            with self.assertRaises(ValueError):validate_positive(e,[bad])
            p.write_text('changed')
            with self.assertRaises(ValueError):validate_positive(e,[decision])

if __name__=='__main__':unittest.main()

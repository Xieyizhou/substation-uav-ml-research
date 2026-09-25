import copy
import tempfile
import unittest
from pathlib import Path
from scripts.vision.record_switchgear_condition_review import validate,file_sha256,OBS_ROWS

class SwitchgearConditionTests(unittest.TestCase):
    def test_all_observations_explicit(self):
        codes=' '.join(OBS_ROWS).split()
        self.assertEqual(len(codes),293)
        self.assertTrue(all(len(c)==2 and c[0] in 'ADLU' and c[1] in 'NPHU' for c in codes))
        self.assertEqual(sum('U' in c for c in codes),27)
    def test_unknown_cannot_be_silently_accepted(self):
        with tempfile.TemporaryDirectory() as temp:
            p=Path(temp)/'evidence';p.write_bytes(b'fixture');h=file_sha256(p)
            r=dict(review_id='T001',bbox_xyxy=[0,0,10,10])
            for k in ('image','label','evidence'):r[k+'_path']=str(p);r[k+'_sha256']=h
            m=dict(items=[r]);d=dict(review_id='T001',bbox_xyxy=r['bbox_xyxy'],review_nature='AI-assisted',reason='Cannot resolve visible panel',reviewed_at='2026-09-08',panel_condition='unknown',occlusion_condition='heavy_foreground_occlusion',decision='held_condition_unknown')
            for k in ('image','label','evidence'):d[k+'_sha256']=h
            validate(m,[d])
            with self.assertRaises(ValueError):validate(m,[d],require_resolved=True)
            bad=copy.deepcopy(d);bad['decision']='condition_recorded'
            with self.assertRaises(ValueError):validate(m,[bad])
            for rows in ([],[d,d]):
                with self.assertRaises(ValueError):validate(m,rows)
            p.write_bytes(b'changed')
            with self.assertRaises(ValueError):validate(m,[d])

if __name__=='__main__':unittest.main()

from copy import deepcopy
from pathlib import Path
import tempfile
import unittest
from scripts.vision.structure_fit import unique,scoring,validate_review,STRUCTURES,file_sha256
from scripts.vision.run_structure_fit import forbidden,cleanup
from scripts.vision.exposure_metrics import missed_reason
from unittest.mock import Mock

class StructureFitTests(unittest.TestCase):
    def test_duplicate_identity(self):
        with self.assertRaises(ValueError):unique([{'id':1},{'id':1}],lambda r:r['id'])
    def test_one_to_one(self):
        t=[dict(class_name='reactor',bbox_xyxy=[0,0,10,10])]
        p=[dict(**t[0],confidence=.8),dict(**t[0],confidence=.7)]
        r=scoring(t,p,p);self.assertEqual(len(r['matches']),1);self.assertEqual(r['unmatched_prediction_count'],1)
    def test_miss_classes(self):
        t=dict(class_name='reactor',bbox_xyxy=[0,0,10,10])
        for ps,want in [([dict(**t,confidence=.2)],'low_confidence_same_class'),
            ([dict(class_name='transformer',bbox_xyxy=[0,0,10,10],confidence=.8)],'wrong_class'),
            ([dict(class_name='reactor',bbox_xyxy=[5,0,15,10],confidence=.8)],'localization'),([], 'no_qualifying_retained_prediction')]:
            self.assertEqual(missed_reason(t,ps),want)
    def test_review_fail_closed(self):
        with tempfile.TemporaryDirectory() as folder:
            ip=Path(folder)/'image';ip.write_bytes(b'original');ep=Path(folder)/'evidence';ep.write_bytes(b'evidence')
            e=dict(event_id='N001',group='negative',source_event_identity='source',image_path=str(ip),image_sha256=file_sha256(ip),evidence_path=str(ep),evidence_sha256=file_sha256(ep))
            d=dict(event_id='N001',source_event_identity='source',image_sha256=e['image_sha256'],evidence_sha256=e['evidence_sha256'],review_nature='AI辅助审核',reason='explicitly viewed',reviewed_at='2026-09-09',structures={k:dict(state='unknown') for k in STRUCTURES})
            evidence={'events':[e]};validate_review(evidence,[d])
            for ds in ([],[d,d],[{**d,'source_event_identity':'changed'}]):
                with self.assertRaises(ValueError):validate_review(evidence,ds)
            b=deepcopy(d);b['structures']['front_panel']={'state':'not_seen'}
            with self.assertRaises(ValueError):validate_review(evidence,[b])
            ip.write_bytes(b'changed')
            with self.assertRaises(ValueError):validate_review(evidence,[d])
    def test_training_guard(self):
        with self.assertRaises(RuntimeError):forbidden()
    def test_already_finished_cleanup(self):
        proc=Mock();proc.poll.return_value=0;cleanup(proc);proc.wait.assert_not_called()

if __name__=='__main__':unittest.main()

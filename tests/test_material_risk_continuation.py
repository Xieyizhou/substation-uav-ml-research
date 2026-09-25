import copy
import tempfile
import unittest
from pathlib import Path
from scripts.vision.contain_material_candidate_risks import require_zero
from scripts.vision.record_material_feasibility_review import validate_decisions
from scripts.vision.continue_material_feasibility_metrics import ratio
from scripts.vision.material_control_feasibility import prior
from src.ml.artifacts import object_sha256

class ContinuationTests(unittest.TestCase):
    def test_zero_risk_exposure_is_required(self):
        require_zero(['safe','safe'],{'held'})
        with self.assertRaises(ValueError):require_zero(['safe','held'],{'held'})

    def evidence(self,folder):
        p=Path(folder)/'image';p.write_bytes(b'bound image bytes')
        sha=prior.file_sha256(p)
        e=dict(evidence_id='nonplanned-target',image_path=str(p),image_sha256=sha,crop_path=str(p),crop_sha256=sha)
        d=dict(evidence_id=e['evidence_id'],evidence_sha256=object_sha256(e),reason='Reviewed own target',page=str(p),page_sha256=sha)
        return e,d,p

    def test_missing_duplicate_and_changed_truth_fail(self):
        with tempfile.TemporaryDirectory() as folder:
            e,d,_=self.evidence(folder)
            validate_decisions([e],[d])
            for decisions in ([],[d,d]):
                with self.assertRaises(ValueError):validate_decisions([e],decisions)
            altered=copy.deepcopy(e);altered['scene_object']='different'
            with self.assertRaises(ValueError):validate_decisions([altered],[d])

    def test_changed_image_and_page_fail(self):
        with tempfile.TemporaryDirectory() as folder:
            e,d,p=self.evidence(folder)
            bad=dict(d,page_sha256='expired')
            with self.assertRaises(ValueError):validate_decisions([e],[bad])
            p.write_bytes(b'changed RGB')
            with self.assertRaises(ValueError):validate_decisions([e],[d])

    def test_zero_denominator_is_unknown(self):
        self.assertIsNone(ratio(0,0));self.assertEqual(ratio(1,2),.5)

if __name__=='__main__':unittest.main()

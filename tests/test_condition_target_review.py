import copy
import tempfile
import unittest
from pathlib import Path
from scripts.vision.record_condition_target_review import (
    TRAIN, DEV, binding, validate, PANEL, OCC, decisions, EVIDENCE_ID, normalize_mapping,
)
from scripts.vision.condition_transfer_design import file_sha256


class ConditionReview(unittest.TestCase):
    def setUp(self):
        self.row=dict(review_id='T001',source=dict(image_sha256='image',label_sha256='label'),
                      evidence_sha256='evidence',target=dict(truth=dict(bbox_xyxy=[1,2,3,4])))
        self.e=dict(training=[self.row],development=[])
        self.d=dict(review_id='T001',binding=binding(self.row),panel_condition=PANEL['U'],
                    occlusion_condition=OCC['U'],decision='held_condition_unknown',
                    pixel_visibility_certified=False,training_admitted=False,promotable=False,
                    review_nature='AI-assisted',reason='Insufficient edge fragment',reviewed_at='2026-09-10T00:00:00Z',
                    full_context_checked=True)

    def test_complete_explicit_observation_count(self):
        self.assertEqual(len(TRAIN),49);self.assertEqual(len(DEV),37)
        self.assertTrue(all(len(c)==4 for c,_,_ in DEV))
        self.assertEqual(sum(c[0]=='U' for c,_ in TRAIN),8)
        self.assertEqual(sum(c.count('U') for c,_,_ in DEV),20)

    def test_unknown_valid_but_not_passed(self):
        validate(self.e,[self.d],False)
        self.d['decision']='condition_recorded'
        with self.assertRaises(ValueError):validate(self.e,[self.d],False)

    def test_missing_duplicate(self):
        for ds in ([],[self.d,self.d]):
            with self.assertRaises(ValueError):validate(self.e,ds,False)

    def test_stale_label_box_and_context_bindings(self):
        for field in ('image_sha256','label_sha256','evidence_sha256','truth'):
            d=copy.deepcopy(self.d);d['binding'][field]='changed'
            with self.assertRaises(ValueError):validate(self.e,[d],False)
        self.row['full_context_sha256']='new-context'
        with self.assertRaises(ValueError):validate(self.e,[self.d],False)

    def test_stale_file(self):
        with tempfile.TemporaryDirectory() as tmp:
            p=Path(tmp)/'evidence';p.write_bytes(b'first')
            self.row.update(evidence_path=str(p),evidence_sha256=file_sha256(p));self.d['binding']=binding(self.row)
            validate(self.e,[self.d]);p.write_bytes(b'changed')
            with self.assertRaises(ValueError):validate(self.e,[self.d])

    def test_no_visibility_or_admission_certification(self):
        for field in ('pixel_visibility_certified','training_admitted','promotable'):
            d=copy.deepcopy(self.d);d[field]=True
            with self.assertRaises(ValueError):validate(self.e,[d],False)

    def test_not_visible_is_not_low_contrast(self):
        self.assertNotEqual(PANEL['N'],PANEL['L'])
        self.d.update(panel_condition=PANEL['N'],occlusion_condition=OCC['N'],decision='condition_recorded')
        validate(self.e,[self.d],False)
        self.assertFalse(self.d['training_admitted'])

    def test_other_evidence_cannot_inherit_observations(self):
        with self.assertRaises(ValueError):decisions(dict(identity='changed',training=[],development=[]))
        with self.assertRaises(ValueError):decisions(dict(identity=EVIDENCE_ID,training=[],development=[]))

    def test_zero_padded_instance_identity(self):
        self.assertEqual(normalize_mapping({'0127':'entry'}),{'127':'entry'})
        with self.assertRaises(ValueError):normalize_mapping({'0127':'a','127':'b'})
        with self.assertRaises(ValueError):normalize_mapping({'unknown':'a'})


if __name__=='__main__':unittest.main()

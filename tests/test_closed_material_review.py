import copy
import unittest
import numpy as np
from scripts.vision.audit_closed_material_mask_coverage import coverage
from scripts.vision.record_closed_material_review import binding,validate,frame_gates


class ClosedReview(unittest.TestCase):
    def setUp(self):
        self.e=dict(event_id='T020-warm-00',source_review_id='T020',variant='warm',source_truth={'annotation_id':'a-instance-0128-box-0'},
                    image_sha256='i',full_context_sha256='f',crop_sha256='c',receipt_identity='r')
        self.q={'events':[self.e]}
        self.d=dict(event_id=self.e['event_id'],binding=binding(self.e),decision='held_content_uncertain',content='insufficient_or_uncertain',
                    review_nature='AI-assisted',reviewed_at='2026-09-10T00:00:00Z',reason='Thin fragment',training_admitted=False,promotable=False,component_pixel_certified=False)

    def test_missing_duplicate(self):
        for ds in ([],[self.d,self.d]):
            with self.assertRaises(ValueError):validate(self.q,ds,False)

    def test_stale_evidence_or_truth(self):
        for k in ('image_sha256','crop_sha256','receipt_identity','source_truth'):
            d=copy.deepcopy(self.d);d['binding'][k]='different'
            with self.assertRaises(ValueError):validate(self.q,[d],False)

    def test_unknown_cannot_pass(self):
        validate(self.q,[self.d],False);self.d['decision']='observed_usable_candidate'
        with self.assertRaises(ValueError):validate(self.q,[self.d],False)

    def test_visible_instance_not_in_boxes(self):
        m=np.zeros((2,2,3),dtype='u1');m[0,0]=[1,0,128]
        self.assertEqual(coverage(m,[],{'128':{}}),[128])
        self.assertEqual(coverage(m,[self.e['source_truth']],{'128':{}}),[])
        with self.assertRaises(ValueError):coverage(m,[],{'127':{}})

    def test_unlabelled_background_not_equipment(self):
        m=np.zeros((2,2,3),dtype='u1');m[0,0]=[0,0,255]
        self.assertEqual(coverage(m,[],{}),[])

    def test_good_existing_labels_do_not_override_unboxed_instance(self):
        self.d.update(decision='observed_usable_candidate',content='clear_body')
        frames=frame_gates(self.q,[self.d],{'anomalies':[dict(source_review_id='T020',variant='warm',runtime_label=128)]})
        f=next(x for x in frames if x['source_review_id']=='T020' and x['variant']=='warm')
        self.assertEqual(f['status'],'held_whole_image');self.assertFalse(f['training_admitted'])


if __name__=='__main__':unittest.main()

import copy,unittest
from unittest.mock import patch
from scripts.vision.inspect_unified_lighting import aligned
from scripts.vision.record_unified_lighting_errors import validate
from scripts.vision.finalize_unified_lighting_evaluation import audit_review,candidate_allowed
from scripts.vision.evaluate_unified_lighting import direct_checks,NAMES,forbidden

class EvaluationTests(unittest.TestCase):
    def frame(self):return dict(pair_id='p',variant='original',image_sha256='i',truth=[dict(annotation_id='a',bbox_xyxy=[0,0,2,2]),dict(annotation_id='b',bbox_xyxy=[2,2,4,4])])
    def test_row_order_not_identity(self):
        a=self.frame();b=copy.deepcopy(a);b['truth'].reverse();aligned([a],[b])
    def test_changed_truth_rejected(self):
        a=self.frame();b=copy.deepcopy(a);b['truth'][0]['bbox_xyxy'][0]=1
        with self.assertRaises(ValueError):aligned([a],[b])
    def test_duplicate_pair_rejected(self):
        with self.assertRaises(ValueError):aligned([self.frame()]*2,[self.frame()])
    def sample(self):
        e=dict(events=[dict(event_id='FP01',kind='FP',predictions=[dict(confidence=.4)],page_sha256='e',source=dict(image_sha256='i'),crops=[dict(sha256='c')])])
        d=dict(decision_id='FP01:0',evidence_sha256='e',image_sha256='i',crop_sha256='c',reason='observed',reviewed_at='now',review_nature='AI-assisted',status='reviewed',prediction=dict(confidence=.4))
        return e,d
    def test_missing_duplicate_review(self):
        e,d=self.sample()
        for ds in ([],[d,d]):
            with self.assertRaises(ValueError):validate(e,ds)
    def test_stale_evidence_rejected(self):
        e,d=self.sample();d['crop_sha256']='old'
        with self.assertRaises(ValueError):validate(e,[d])
    def test_changed_prediction_rejected(self):
        e,d=self.sample();d['prediction']['confidence']=.9
        with self.assertRaises(ValueError):audit_review(e,dict(decisions=[d]))
    def test_pending_blocks_candidate(self):
        s=dict(matching_conflicts=0,policy_results={'L-physical':dict(passed=True)},direct_retention=dict(passed=True))
        self.assertTrue(candidate_allowed(s,[]));self.assertFalse(candidate_allowed(s,['unknown']))
        s['matching_conflicts']=1;self.assertFalse(candidate_allowed(s,[]))
    def test_direct_recall_tolerance(self):
        def group(v):return {c:dict(instance_recall=dict(mean=v),per_class={n:dict(instance_recall=dict(mean=v)) for n in NAMES}) for c in ('original','lighting')}
        self.assertTrue(direct_checks(group(.75),group(.8))['passed'])
        self.assertFalse(direct_checks(group(.749),group(.8))['passed'])
    def test_training_forbidden(self):
        with self.assertRaises(AssertionError):forbidden()

if __name__=='__main__':unittest.main()

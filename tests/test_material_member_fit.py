import copy
import unittest
from scripts.vision.material_member_fit import scoring
from scripts.vision.infer_material_member_fit import validate_unit,forbidden
from scripts.vision.review_material_member_fit import validate
from src.ml.artifacts import object_sha256

def signed(x):return dict(x,identity=object_sha256(x))

class FitTests(unittest.TestCase):
    def fixture(self):
        t=dict(class_name='reactor',bbox_xyxy=[0,0,20,20],annotation_id='a',label_line_index=0)
        pred=dict(class_name='reactor',bbox_xyxy=[0,0,20,20],confidence=.8)
        m=dict(member_id='m',truth=[t],image_sha256='image')
        p=dict(identity='protocol',members=[m],models={'model':dict(counts={'m':3})})
        row=dict(member_id='m',image_sha256='image',actual_exposures=3,**scoring([t],[pred],[pred]))
        r=dict(status='complete',protocol_identity='protocol',model='model',rows=[row],inputs={},
            optimizer_created=False,backward_executed=False,training_validation_executed=False)
        return p,r

    def test_full_matching_exposure_and_members(self):
        p,r=self.fixture();validate_unit(signed(r),'model',p)
        for mutate in ('missing','duplicate','exposure','truth','matching','runtime'):
            x=copy.deepcopy(r)
            if mutate=='missing':x['rows']=[]
            if mutate=='duplicate':x['rows']*=2
            if mutate=='exposure':x['rows'][0]['actual_exposures']=0
            if mutate=='truth':x['rows'][0]['truth'][0]['bbox_xyxy'][0]=1
            if mutate=='matching':x['rows'][0]['matches']=[]
            if mutate=='runtime':x['optimizer_created']=True
            with self.assertRaises(ValueError):validate_unit(signed(x),'model',p)

    def test_review_missing_duplicate_expired(self):
        e=dict(event_id='own',truth={'class':'reactor'})
        p=dict(identity='p',events=[e]);d=dict(event_id='own',evidence_sha256=object_sha256(e))
        body=dict(protocol_identity='p',decisions=[d],inputs={});validate(p,signed(body))
        for ds in ([],[d,d],[dict(d,evidence_sha256='old')]):
            with self.assertRaises(ValueError):validate(p,signed(dict(body,decisions=ds)))

    def test_training_forbidden(self):
        with self.assertRaises(RuntimeError):forbidden()

    def test_duplicate_prediction_not_double_recall(self):
        p,r=self.fixture();row=r['rows'][0]
        result=scoring(row['truth'],row['predictions']*2,row['low_predictions'])
        self.assertEqual(len(result['matches']),1);self.assertEqual(result['unmatched_prediction_count'],1)

if __name__=='__main__':unittest.main()

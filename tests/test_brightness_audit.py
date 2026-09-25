import unittest,copy
from scripts.vision.audit_brightness_transfer import check_factors,validate_review
from scripts.vision.brightness_transfer_runtime import factor

class AuditTests(unittest.TestCase):
    def test_exact_factors_and_sequence(self):
        p={'schedules':{},'brightness_factors':{}}
        for s in (7,17,27):
            for a in ('noaug','brightness'):
                k=f'{a}-450-{s}';p['schedules'][k]=['m']*2700;p['brightness_factors'][k]=[1. if a=='noaug' else factor(s,'m',i) for i in range(2700)]
        check_factors(p)
        q=copy.deepcopy(p);q['brightness_factors']['brightness-450-7'][0]=.799
        with self.assertRaises(ValueError):check_factors(q)
        q=copy.deepcopy(p);q['schedules']['brightness-450-7'][0]='other'
        with self.assertRaises(ValueError):check_factors(q)
    def fixture(self):
        r={'event_id':'FP01','kind':'FP','predictions':[{'x':1}],'source':{'image_sha256':'i'},'evidence_sha256':'e','crops':[{'sha256':'c'}]}
        d={'review_id':'FP01:0','evidence_sha256':'e','image_sha256':'i','crop_sha256':'c','review_nature':'AI辅助审核','reviewed_at':'now','reason':'observed','pixel_visibility_certified':False,'prediction':{'x':1}}
        return {'events':[r]},d
    def test_missing_duplicate_review(self):
        e,d=self.fixture();validate_review(e,[d])
        for ds in ([],[d,d]):
            with self.assertRaises(ValueError):validate_review(e,ds)
    def test_stale_or_changed_prediction(self):
        e,d=self.fixture()
        for k,v in [('crop_sha256','stale'),('prediction',{'x':2}),('pixel_visibility_certified',True)]:
            with self.assertRaises(ValueError):validate_review(e,[{**d,k:v}])

if __name__=='__main__':unittest.main()

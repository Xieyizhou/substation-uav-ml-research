import copy,unittest
from collections import Counter
from unittest.mock import patch
from scripts.vision.physical_low_light_design import tails
from scripts.vision.physical_low_light_dataset import validate_review

class PhysicalDesignTests(unittest.TestCase):
    def test_exact_tail_pairing(self):
        rows=[dict(planned_category=c,member_id=f'low-{c}-{i}',source_member_id=f'{c}-{i}') for c in ('transformer','switchgear','capacitor_bank','reactor') for i in range(8)]
        for seed in (7,17,27):
            r,l,b=tails(seed,rows)
            self.assertEqual((r,l,b),tails(seed,list(reversed(rows))))
            self.assertEqual(len(r),600);self.assertEqual(len(b),100)
            self.assertEqual(sum(x!=y for x,y in zip(r,l)),300)
            low=Counter(x.removeprefix('low-') for x in l if x.startswith('low-'));normal=Counter(x for x in l if not x.startswith('low-'))
            self.assertEqual(low,normal);self.assertEqual(Counter(r),low+normal)
            self.assertEqual(set(low),{x['source_member_id'] for x in rows})
            for i in range(0,100,2):
                self.assertEqual(r[i*6:(i+1)*6],r[(i+1)*6:(i+2)*6])
                self.assertEqual({b[i]['low_light'],b[i+1]['low_light']},{True,False})
            for i in (0,50):self.assertEqual(sum(x['low_light'] for x in b[i:i+50]),25)

    def test_missing_source_blocks(self):
        with self.assertRaises(ValueError):tails(7,[])

    def test_review_rejects_unknown_missing_duplicate_stale(self):
        p={'units':[{'unit_id':'L01'}]};e={'identity':'e','events':[{'event_id':'L01:a','crop_sha256':'crop'}]}
        d=dict(event_id='L01:a',evidence_identity='e',crop_sha256='crop',status='content_sufficient_for_bounded_research',reason='observed',review_nature='AI辅助审核',training_admitted=False,promotable=False)
        r=dict(full_frames_viewed=['L01'],decisions=[d])
        with patch('scripts.vision.physical_low_light_dataset.prior.verify'),patch('scripts.vision.physical_low_light_dataset.evidence',return_value=e):
            validate_review(p,r)
            for kind in ('unknown','missing','duplicate','stale','no_full_frame'):
                bad=copy.deepcopy(r)
                if kind=='unknown':bad['decisions'][0]['status']='unknown'
                if kind=='missing':bad['decisions']=[]
                if kind=='duplicate':bad['decisions']*=2
                if kind=='stale':bad['decisions'][0]['evidence_identity']='changed'
                if kind=='no_full_frame':bad['full_frames_viewed']=[]
                with self.assertRaises(ValueError):validate_review(p,bad)

if __name__=='__main__':unittest.main()

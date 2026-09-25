import copy
import unittest
from scripts.vision.record_closed_budget_review import DEST,prior,validate
from scripts.vision.evaluate_closed_budget import OUT,KEYS,validate_record,verify_binding

class BudgetEvaluationTests(unittest.TestCase):
    def test_all_six_bindings(self):
        for key in KEYS:
            r=prior.read(OUT/'evaluation'/f'{key}.json');validate_record(r,key);verify_binding(r,key)
            self.assertEqual(r['effective_cpu_threads'],4)
            self.assertEqual(r['matching_conflicts'],0)
    def test_explicit_review_missing_duplicate_stale_unknown(self):
        e=prior.read(DEST/'evidence.json');r=prior.read(DEST/'review.json');prior.verify(e);prior.verify(r);validate(e,r['decisions'])
        self.assertEqual(len(r['decisions']),48);self.assertEqual(len(r['pending_ids']),5)
        for mode in ('missing','duplicate','hash','unknown_pass','truth'):
            ds=copy.deepcopy(r['decisions'])
            if mode=='missing':ds.pop()
            if mode=='duplicate':ds[-1]=ds[0]
            if mode=='hash':ds[0]['evidence_sha256']='changed'
            if mode=='unknown_pass':next(x for x in ds if x['content']=='unknown_instance_content')['status']='observed_not_admitted'
            if mode=='truth':ds[-1]['truth']['bbox_xyxy'][0]+=1
            with self.subTest(mode=mode),self.assertRaises(ValueError):validate(e,ds)
    def test_transitions_and_competition_preserved(self):
        e=prior.read(DEST/'evidence.json');self.assertEqual(len(e['transitions']),720)
        a=prior.read(OUT/'evaluation/audit.json');prior.verify(a)
        self.assertEqual(len(a['matching_competitions']),3)
        self.assertTrue(all(v['first_450_losses_exact'] for v in a['prefix_checks'].values()))
        self.assertIsNone(a['selected_candidate'])

if __name__=='__main__':unittest.main()

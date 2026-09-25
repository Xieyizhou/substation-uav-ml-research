import copy,unittest
from unittest.mock import patch
from scripts.vision.closed_gamma_design import OUT,KEYS,freeze,prior
from scripts.vision.preflight_closed_gamma import receipt
from scripts.vision.review_closed_gamma import validate

class GammaReadinessTests(unittest.TestCase):
    def test_real_all_loaders(self):
        p=freeze()
        for key in KEYS:
            r,_=receipt(key,p)
            self.assertEqual(len(r['actual']),5400)
            self.assertEqual(len(r['batch_records']),900)
            self.assertTrue(r['identity_batches_exact'] and r['full_supervision_exact'])
            self.assertFalse(r['optimizer_created'] or r['backward_executed'] or r['validation_run'])

    def test_review_semantic_fail_closed(self):
        e=prior.read(OUT/'augmentation-review/evidence.json');r=prior.read(OUT/'augmentation-review/review.json');validate(e,r)
        # Bypass signature only to independently exercise semantic rejection.
        for mutation in ('missing','duplicate','stale','unknown'):
            bad=copy.deepcopy(r)
            if mutation=='missing':bad['decisions'].pop()
            elif mutation=='duplicate':bad['decisions'][-1]=bad['decisions'][0]
            elif mutation=='stale':bad['decisions'][0]['evidence']['card_sha256']='changed'
            else:bad['decisions'][0]['status']='unknown'
            with patch.object(prior,'verify'),self.assertRaises(ValueError):validate(e,bad)

    def test_review_signature(self):
        e=prior.read(OUT/'augmentation-review/evidence.json');r=prior.read(OUT/'augmentation-review/review.json')
        r['decisions'][0]['reason']='changed'
        with self.assertRaises(ValueError):validate(e,r)

    def test_actual_entry_probes(self):
        for mode in ('identity','gamma'):
            r=prior.read(OUT/'entry-probes'/mode/'completion.json');prior.verify(r)
            self.assertEqual(r['result']['optimizer_steps'],10)
            self.assertEqual(len(r['result']['actual']),60)
            self.assertEqual(r['result']['effective_threads'],4)
            self.assertTrue(r['result']['probe_only'])

if __name__=='__main__':unittest.main()

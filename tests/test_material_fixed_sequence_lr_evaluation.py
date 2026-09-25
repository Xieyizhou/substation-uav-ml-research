import unittest
from unittest.mock import patch
from scripts.vision import evaluate_material_fixed_sequence_lr as e

class EvaluationTests(unittest.TestCase):
    def test_default_never_infers(self):
        with patch('sys.argv',['evaluate']), patch.object(e,'complete') as c, patch.object(e,'evaluate') as run:
            e.main()
        self.assertEqual(c.call_count,3)
        run.assert_not_called()

    def test_training_forbidden(self):
        with self.assertRaises(AssertionError): e.forbidden()

    def test_weight_protocol_and_runtime_bound(self):
        c=dict(weights='weight',weights_sha256='w')
        r=dict(inputs={'weight':'w',str(e.OUT/'protocol.json'):'p'},optimizer_created=False,backward_executed=False,validation_run=False)
        with patch.object(e.prior,'read',return_value=c),patch.object(e.prior,'file_sha256',return_value='p'):
            e.verify_binding(r,'Q-7')
            for field in ('weight','protocol','runtime'):
                x=dict(r,inputs=dict(r['inputs']))
                if field=='weight':x['inputs']['weight']='old'
                elif field=='protocol':x['inputs'][str(e.OUT/'protocol.json')]='old'
                else:x['optimizer_created']=True
                with self.assertRaises(ValueError):e.verify_binding(x,'Q-7')

    def test_incomplete_and_duplicate_rejected(self):
        r=dict(status='complete',cell='Q-7',rows=[],negative_rows=[])
        with patch.object(e.prior,'verify'):
            with self.assertRaises(ValueError): e.validate_record(r,'Q-7')
            r.update(rows=[dict(pair_id='one',variant='original')]*48,
                     negative_rows=[dict(view_id='one',variant='original')]*48)
            with self.assertRaises(ValueError): e.validate_record(r,'Q-7')

    def test_stale_identity_rejected(self):
        with patch.object(e.prior,'verify',side_effect=ValueError('stale')):
            with self.assertRaises(ValueError): e.validate_record({},'Q-7')

if __name__=='__main__': unittest.main()

import unittest
from pathlib import Path
from scripts.vision.exposure_protocol import read

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'data/research/ml_training_recovery_v1/hard-negative-coverage-v1/exposure-order-diagnosis-v1/permutation-root-cause-v1'

class RootCauseEvidenceTests(unittest.TestCase):
    def test_input_forward_assignment_identity(self):
        r=read(OUT/'single-batch.json')
        self.assertEqual(r['input_checks']['images']['changed'],0)
        for row in r['input_checks']['targets']:
            self.assertTrue(all(v['changed']==0 for v in row.values()))
        for mode in ('eval','train','frozen_bn'):
            for field in ('layers','outputs','assignments','buffers'):
                self.assertTrue(all(v['changed']==0 for v in r['results'][mode][field].values()))
    def test_reduction_and_later_assignment_divergence(self):
        r=read(OUT/'trace.json');a,b=r['raw'][0]['target_score_sums']
        self.assertNotEqual(a['fp32'],b['fp32']);self.assertEqual(a['fp64'],b['fp64'])
        self.assertEqual(next(x['step'] for x in r['raw'] if x['assignment_differences']['/3']['changed']),11)
        self.assertTrue(all(x['gradient_max']==x['state_max']==0 for x in r['canonical']))
    def test_adam_first_step_formula(self):
        r=read(OUT/'first-adam-update.json');top=r['top_differences'][0]
        self.assertLess(top['gradient_F']*top['gradient_W'],0)
        for index,suffix in enumerate(('F','W')):
            grad=top['gradient_'+suffix]*r['clip_scale'][index]
            expected=top['initial_value']-.001*grad/(abs(grad)+r['adam_epsilon'])
            self.assertAlmostEqual(expected,top['value_'+suffix],delta=5e-7)
    def test_canonical_full_plan(self):
        from scripts.vision.canonical_batch_order import canonicalize
        base=OUT.parent
        f=read(base.parent/'fixed-sequence-seed-diagnosis-v1/protocol.json')['schedules']['F-100-7']
        w=read(base/'protocol.json')['schedules']['W-100-7']
        self.assertNotEqual(f,w);self.assertEqual(canonicalize(f),canonicalize(w))

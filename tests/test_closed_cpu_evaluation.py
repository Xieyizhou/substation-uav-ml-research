import copy
import unittest
from scripts.vision.benchmark_closed_cpu import OUT,prior,CONFIGS
from scripts.vision.evaluate_closed_source_control import validate_record


class ClosedCpuEvaluationTests(unittest.TestCase):
    def test_selected_exact_and_fastest(self):
        r=prior.read(OUT/'completion.json');prior.verify(r)
        eligible=[x for x in r['results'] if x['exact_predictions']]
        self.assertEqual(r['selected'],min(eligible,key=lambda x:x['wall_seconds'])['config'])
        self.assertEqual(r['selected'],'dual4')

    def test_real_threads_and_repeat_equivalence(self):
        for config,(threads,_) in CONFIGS.items():
            for key in ('E-7','M-7'):
                r=prior.read(OUT/config/(key+'.json'));prior.verify(r)
                self.assertEqual(r['threads'],threads)
                self.assertEqual(len(r['outputs']),3)
                self.assertTrue(all(x==r['outputs'][0] for x in r['outputs']))
                self.assertFalse(r['optimizer_created'])
                self.assertFalse(r['backward_executed'])
                self.assertFalse(r['validation_run'])
        for key in ('E-7','M-7'):
            self.assertEqual(prior.read(OUT/'single4'/(key+'.json'))['outputs'],prior.read(OUT/'dual4'/(key+'.json'))['outputs'])

    def test_modified_benchmark_rejected(self):
        r=copy.deepcopy(prior.read(OUT/'completion.json'));r['selected']='single8'
        with self.assertRaises(ValueError):prior.verify(r)


if __name__=='__main__':unittest.main()

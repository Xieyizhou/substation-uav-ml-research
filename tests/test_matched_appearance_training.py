import copy
import subprocess
import tempfile
import unittest
from pathlib import Path
from unittest.mock import Mock, patch
from scripts.vision import run_matched_appearance_training as run

class MatchedTrainingTests(unittest.TestCase):
    def args(self):
        return dict(imgsz=640,batch=6,nbs=6,device='cpu',workers=0,optimizer='AdamW',lr0=.001,lrf=1.,
            warmup_epochs=0,epochs=30,amp=False,deterministic=True,val=False,**{k:0 for k in run.ZERO_AUG})

    def test_controls_reject_lower_rate_and_augmentation(self):
        run.validate_args(self.args())
        for k,v in [('lr0',.0003),('warmup_epochs',1),('batch',12),('nbs',64),('epochs',29),('device','mps'),('hsv_v',.1)]:
            a=self.args();a[k]=v
            with self.assertRaises(ValueError):run.validate_args(a)

    def record(self):
        return dict(status='complete',cell='K-300-7',matching_conflicts=0,
            rows=[dict(pair_id=i,variant=v,matching_conflict=False) for i in range(12) for v in range(4)],
            negative_rows=[dict(view_id=i,variant='original') for i in range(48)])

    def test_incomplete_duplicate_conflicting_evaluation(self):
        run.validate_evaluation(self.record(),'K-300-7')
        for kind in ('missing','duplicate','negative_duplicate','conflict','wrong_cell'):
            r=self.record()
            if kind=='missing':r['rows'].pop()
            if kind=='duplicate':r['rows'][1]=r['rows'][0]
            if kind=='negative_duplicate':r['negative_rows'][1]=r['negative_rows'][0]
            if kind=='conflict':r['rows'][0]['matching_conflict']=True
            if kind=='wrong_cell':r['cell']='L-300-7'
            with self.assertRaises(ValueError):run.validate_evaluation(r,'K-300-7')

    def test_retention_boundary_and_per_class_failure(self):
        def group(value):
            return {v:dict(instance_recall=dict(mean=value),per_class={n:dict(instance_recall=dict(mean=value)) for n in run.NAMES}) for v in ('original','lighting')}
        self.assertTrue(all(r['passed'] for r in run.retention(group(.75),group(.8),.05)))
        a=group(.8);a['lighting']['per_class']['reactor']['instance_recall']['mean']=.749
        result=run.retention(a,group(.8),.05)
        self.assertEqual(sum(not r['passed'] for r in result),1)

    def test_worker_attempts_bounded(self):
        with tempfile.TemporaryDirectory() as d, patch.object(run,'OUT',Path(d)):
            for n in (1,2,3):(Path(d)/'execution-attempts/K-300-7'/f'attempt-{n:03}').mkdir(parents=True)
            with self.assertRaises(ValueError):run.run_worker('K-300-7')

    def test_cleanup_term_then_kill(self):
        proc=Mock(pid=123);proc.poll.return_value=None
        proc.wait.side_effect=[subprocess.TimeoutExpired('worker',10),0]
        with patch.object(run.os,'killpg') as kill:
            run.stop_process(proc)
            self.assertEqual(kill.call_count,2)
            self.assertEqual(proc.wait.call_count,2)

    def test_worker_failure_preserved_and_cleanup(self):
        proc=Mock(pid=123);proc.wait.return_value=1;proc.poll.return_value=1
        with tempfile.TemporaryDirectory() as d, patch.object(run,'OUT',Path(d)), patch.object(run.subprocess,'Popen',return_value=proc):
            with self.assertRaises(RuntimeError):run.run_worker('K-300-7')
            failure=run.read(Path(d)/'execution-attempts/K-300-7/attempt-001/failure.json')
            self.assertTrue(failure['process_cleanup_confirmed']);self.assertFalse(failure['training_admitted'])

    def test_cancel_preserves_failure(self):
        proc=Mock(pid=123);proc.wait.side_effect=KeyboardInterrupt;proc.poll.return_value=0
        with tempfile.TemporaryDirectory() as d, patch.object(run,'OUT',Path(d)), patch.object(run.subprocess,'Popen',return_value=proc):
            with self.assertRaises(KeyboardInterrupt):run.run_worker('L-300-7')
            self.assertTrue((Path(d)/'execution-attempts/L-300-7/attempt-001/failure.json').exists())

if __name__=='__main__':unittest.main()

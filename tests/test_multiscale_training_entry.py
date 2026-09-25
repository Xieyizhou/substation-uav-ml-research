import unittest
from unittest.mock import patch,Mock
import torch
from types import SimpleNamespace
from scripts.vision import train_frozen_multiscale as entry
from scripts.vision.preflight_unified_lighting import tensor_hash
from scripts.vision.frozen_multiscale_runtime import preprocess

class EntryTests(unittest.TestCase):
    def fixture(self):
        torch.set_num_threads(2)
        owner=SimpleNamespace(device=torch.device('cpu'),args=SimpleNamespace(multi_scale=0),stride=32)
        batch=dict(img=torch.zeros((6,3,640,640),dtype=torch.uint8),im_file=['x']*6,cls=torch.tensor([[3.]]),bboxes=torch.tensor([[.5,.5,.2,.3]]),batch_idx=torch.tensor([0.]))
        out=preprocess(owner,batch,320)
        record=dict(members=['m']*6,size=320,source640_sha256=tensor_hash(batch['img']),full_supervision={f:tensor_hash(batch[f]) for f in ('cls','bboxes','batch_idx')},effective_shape=list(out['img'].shape),effective_tensor_sha256=tensor_hash(out['img']))
        return owner,batch,dict(batches=[record],actual=['m']*6,brightness_log=[])
    def test_real_hook_accepts_expected_tensor(self):
        owner,b,e=self.fixture();g=entry.BatchGate(e,{'x':'m'})
        self.assertEqual(tuple(g.apply(owner,b)['img'].shape),(6,3,320,320));self.assertEqual(g.step,1)
        with self.assertRaises(ValueError):g.finish([])
    def test_changed_member_and_tensor_fail(self):
        for kind in ('member','image','label','output'):
            owner,b,e=self.fixture();g=entry.BatchGate(e,{'x':'wrong' if kind=='member' else 'm'})
            if kind=='image':b['img'][0,0,0,0]=1
            if kind=='label':b['bboxes'][0,0]=.4
            if kind=='output':e['batches'][0]['effective_tensor_sha256']='expired'
            with self.assertRaises(ValueError):g.apply(owner,b)
    def test_default_and_worker_guard(self):
        with patch.object(entry,'ready') as ready,patch.object(entry,'run') as run,patch.object(entry,'worker') as worker:
            entry.main([]);ready.assert_called_once();run.assert_not_called();worker.assert_not_called()
            with self.assertRaises(SystemExit):entry.main(['--worker','fixed-7'])
            worker.assert_not_called()
    def test_finished_process_not_signalled(self):
        proc=Mock();proc.poll.return_value=0
        with patch.object(entry.os,'killpg') as kill:entry.cleanup(proc);kill.assert_not_called()
    def test_cancelled_process_is_cleaned(self):
        proc=Mock(pid=123);proc.poll.return_value=None
        with patch.object(entry.os,'killpg') as kill:entry.cleanup(proc);kill.assert_called_once_with(123,entry.signal.SIGTERM);proc.wait.assert_called_once()

if __name__=='__main__':unittest.main()

import unittest
from types import SimpleNamespace
from unittest.mock import patch
import torch
from scripts.vision.train_gray_body_control import BatchGate,main,tensor_hash

class EntryTests(unittest.TestCase):
    def sample(self):
        batch=dict(img=torch.zeros((6,3,640,640),dtype=torch.uint8),cls=torch.zeros((1,1)),bboxes=torch.ones((1,4))*.2,batch_idx=torch.zeros(1),im_file=['a']*6)
        record=dict(members=['member']*6,image_tensor_sha256=tensor_hash(batch['img']),full_supervision={k:tensor_hash(batch[k]) for k in ('cls','bboxes','batch_idx')})
        return batch,dict(batch_records=[record],actual=['member']*6,brightness_log=[])
    def test_batch_and_fixed_preprocess(self):
        b,e=self.sample();g=BatchGate(e,{'a':'member'})
        out=g.apply(SimpleNamespace(device=torch.device('cpu'),args=SimpleNamespace(multi_scale=0)),b)
        self.assertEqual(g.actual,['member']*6);self.assertEqual(out['img'].dtype,torch.float32)
    def test_changed_tensor_stops(self):
        b,e=self.sample();b['img'][0,0,0,0]=1
        with self.assertRaises(ValueError):BatchGate(e,{'a':'member'}).apply(None,b)
    def test_changed_labels_stops(self):
        b,e=self.sample();b['cls'][0,0]=1
        with self.assertRaises(ValueError):BatchGate(e,{'a':'member'}).apply(None,b)
    def test_default_never_trains(self):
        with patch('scripts.vision.train_gray_body_control.ready') as ready,patch('scripts.vision.train_gray_body_control.run',side_effect=AssertionError('Training')):
            main([]);ready.assert_called_once()
    def test_incomplete_cannot_finish(self):
        _,e=self.sample()
        with self.assertRaises(ValueError):BatchGate(e,{}).finish([])

if __name__=='__main__':unittest.main()

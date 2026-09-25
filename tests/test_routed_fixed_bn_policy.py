import unittest
import torch
from scripts.vision import routed_fixed_bn_policy as p

class PolicyTest(unittest.TestCase):
    def model(self):
        m=torch.nn.Module();m.model=torch.nn.ModuleList([
            torch.nn.Sequential(torch.nn.Conv2d(3,3,1),torch.nn.BatchNorm2d(3)) for _ in range(12)])
        return m
    def test_buffers_fixed_affine_learns_neck_updates(self):
        m=self.model();initial=p.digest(m);m.train();p.apply(m)
        x=torch.randn(2,3,8,8)
        loss=m.model[0](x).square().mean()+m.model[11](x).square().mean()
        loss.backward();p.check(m,initial)
        self.assertIsNotNone(m.model[0][0].weight.grad)
        self.assertIsNotNone(m.model[0][1].weight.grad)
        self.assertEqual(m.model[0][1].num_batches_tracked.item(),0)
        self.assertEqual(m.model[11][1].num_batches_tracked.item(),1)
    def test_mode_or_buffer_drift_rejected(self):
        m=self.model();initial=p.digest(m);p.apply(m)
        m.model[0][1].train()
        with self.assertRaises(ValueError):p.check(m,initial)
        p.apply(m);m.model[0][1].running_mean.add_(1)
        with self.assertRaises(ValueError):p.check(m,initial)
    def test_copy_only_buffers_not_parameters(self):
        m=self.model();ema=self.model();before=ema.model[0][0].weight.clone()
        p.copy_buffers(m,ema)
        self.assertEqual(p.digest(m),p.digest(ema))
        self.assertTrue(torch.equal(before,ema.model[0][0].weight))

if __name__=='__main__':unittest.main()

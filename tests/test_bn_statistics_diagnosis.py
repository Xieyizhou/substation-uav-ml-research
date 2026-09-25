import unittest
import torch
from scripts.vision.diagnose_bn_statistics import exchange

class BufferGate(unittest.TestCase):
    def models(self):
        return (torch.nn.Sequential(*(torch.nn.BatchNorm2d(1) for _ in range(81))) for _ in range(2))
    def test_only_statistics(self):
        a,b=self.models()
        with torch.no_grad():
            for m in b:
                m.running_mean.fill_(2);m.running_var.fill_(3);m.weight.fill_(7);m.num_batches_tracked.fill_(12)
        names=exchange(a,b)
        self.assertEqual(len(names),162)
        for m in a:
            self.assertEqual(m.running_mean.item(),2);self.assertEqual(m.running_var.item(),3)
            self.assertEqual(m.weight.item(),1);self.assertEqual(m.num_batches_tracked.item(),0)
    def test_fused_rejected(self):
        a,b=self.models()
        with self.assertRaises(ValueError):exchange(torch.nn.Identity(),b)
    def test_nonfinite_rejected(self):
        a,b=self.models();b[0].running_mean.fill_(float('nan'))
        with self.assertRaises(ValueError):exchange(a,b)

if __name__=='__main__':unittest.main()

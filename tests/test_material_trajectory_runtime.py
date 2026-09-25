import copy
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
import torch
from scripts.vision.material_trajectory_runtime import capture,state_digest,rng_digest


class CaptureTests(unittest.TestCase):
    def test_capture_no_mutation_and_reload(self):
        model=torch.nn.Sequential(torch.nn.Linear(3,4),torch.nn.BatchNorm1d(4))
        owner=SimpleNamespace(model=model,ema=SimpleNamespace(ema=copy.deepcopy(model),updates=3),
                              args=SimpleNamespace(imgsz=640,device='cpu'))
        model(torch.ones(2,3)).sum().backward()
        before=(state_digest(model),rng_digest());grads=[p.grad.clone() for p in model.parameters()]
        with tempfile.TemporaryDirectory() as d:
            path=Path(d)/'step.pt';r=capture(owner,path,3)
            self.assertTrue(r['observation_state_unchanged'])
            loaded=torch.load(path,weights_only=False)
            self.assertEqual(state_digest(loaded['model']),before[0])
            self.assertEqual((state_digest(model),rng_digest()),before)
            for p,g in zip(model.parameters(),grads):self.assertTrue(torch.equal(p.grad,g))
            with self.assertRaises(ValueError):capture(owner,path,3)


if __name__=='__main__':unittest.main()

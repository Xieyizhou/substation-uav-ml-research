import copy
import unittest
from types import SimpleNamespace
import torch
from ultralytics.models.yolo.detect import DetectionTrainer
from tests.test_routed_backbone_control import Model
from scripts.vision import routed_small_backbone_control as arm

class RegionalTests(unittest.TestCase):
    def test_real_optimizer_scheduler_and_updates(self):
        m=Model();t=SimpleNamespace(args=SimpleNamespace())
        opt=DetectionTrainer.build_optimizer(t,m,name='AdamW',lr=.00025,decay=.0005)
        original={id(p):g['weight_decay'] for g in opt.param_groups for p in g['params']}
        opt.param_groups[:]=arm.split_groups(m,opt.param_groups)
        sched=torch.optim.lr_scheduler.LambdaLR(opt,lambda epoch:1.)
        before=[p.detach().clone() for p in m.parameters()]
        for _ in range(2):
            arm.assert_optimizer(m,opt)
            opt.zero_grad();sum(p.sum() for p in m.parameters()).backward();opt.step();sched.step()
        self.assertTrue(all(not torch.equal(a,b) for a,b in zip(before,m.parameters())))
        self.assertEqual(original,{id(p):g['weight_decay'] for g in opt.param_groups for p in g['params']})
        arm.assert_optimizer(m,opt)
    def test_duplicate_and_missing_parameters_rejected(self):
        m=Model();params=list(m.parameters())
        for ps in (params[:-1],params+[params[0]]):
            with self.assertRaises(ValueError):arm.split_groups(m,[dict(params=ps,lr=.00025)])
    def test_wrong_lr_and_frozen_bn_rejected(self):
        m=Model();groups=arm.split_groups(m,[dict(params=list(m.parameters()),lr=.00025)])
        opt=SimpleNamespace(param_groups=groups);arm.assert_optimizer(m,opt)
        groups[0]['lr']=.00025
        with self.assertRaises(ValueError):arm.assert_optimizer(m,opt)
        groups[0]['lr']=.000025;m.model[0][1].eval()
        with self.assertRaises(ValueError):arm.assert_optimizer(m,opt)

if __name__=='__main__':unittest.main()

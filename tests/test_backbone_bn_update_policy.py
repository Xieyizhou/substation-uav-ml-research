import copy
import unittest
import torch
from tests.test_routed_backbone_control import frozen_model
from scripts.vision import backbone_bn_update_policy as policy
from scripts.vision.routed_backbone_control import state_digest

class BNPolicyTests(unittest.TestCase):
    def test_real_forward_updates_only_allowed_buffers(self):
        m=frozen_model();policy.activate(m);fixed=policy.digest(m);mutable=policy.digest(m,True)
        with torch.no_grad():m.model(torch.randn(2,3,8,8))
        self.assertEqual(fixed,policy.digest(m));self.assertNotEqual(mutable,policy.digest(m,True))
        self.assertEqual(m.model[0][1].num_batches_tracked.item(),1)
        policy.assert_policy(m)

    def test_silent_eval_and_parameter_unfreeze_rejected(self):
        m=frozen_model()
        with self.assertRaises(ValueError):policy.assert_policy(m)
        policy.activate(m);next(m.parameters()).requires_grad_(True)
        with self.assertRaises(ValueError):policy.assert_policy(m)

    def test_ema_copies_bn_and_parameters_without_touching_head(self):
        m=frozen_model();policy.activate(m);ema=copy.deepcopy(m)
        with torch.no_grad():m.model(torch.randn(2,3,8,8))
        head=state_digest(ema,False);policy.sync_ema(m,ema)
        self.assertEqual(head,state_digest(ema,False))
        self.assertEqual(ema.model[0][1].num_batches_tracked.item(),1)

    def test_parameter_drift_is_not_allowed_bn_change(self):
        m=frozen_model();before=policy.digest(m)
        with torch.no_grad():m.model[0][1].weight.add_(1)
        self.assertNotEqual(before,policy.digest(m))

if __name__=='__main__':unittest.main()

import copy
import unittest
from scripts.vision import routed_backbone_bn_control as arm

class ProtocolTests(unittest.TestCase):
    def pair(self):
        s={f:[] for f in ('pool_rows','names','initialization','evaluation','environment','held_members','training_config')}
        s['configuration']={'lr':.00025}
        for f in ('schedules','exposures','windows','listings','contrast'):s[f]={k:[1,2] for k in arm.SOURCE_KEYS}
        p=copy.deepcopy(s)
        for f in ('schedules','exposures','windows','listings','contrast'):p[f]={k:s[f][o][:] for k,o in zip(arm.KEYS,arm.SOURCE_KEYS)}
        p['configuration']['parameter_update_policy']=arm.policy.VERSION
        p['backbone_policy']=dict(version=arm.policy.VERSION,indices=arm.old.FROZEN,batchnorm='train_running_stats_update_affine_fixed',ema='copy_entire_backbone_exactly_after_each_update',other_parameters='train_except_existing_dfl')
        return s,p
    def test_only_policy_changes(self):
        s,p=self.pair();arm.validate(s,p)
        for f in ('pool_rows','initialization','evaluation','training_config'):
            q=copy.deepcopy(p);q[f]=['different']
            with self.assertRaises(ValueError):arm.validate(s,q)
        for f in ('schedules','exposures','windows','listings','contrast'):
            q=copy.deepcopy(p);q[f][arm.KEYS[0]][0]=9
            with self.assertRaises(ValueError):arm.validate(s,q)
    def test_ema_rule_is_frozen(self):
        s,p=self.pair();p['backbone_policy']['ema']='average'
        with self.assertRaises(ValueError):arm.validate(s,p)

if __name__=='__main__':unittest.main()

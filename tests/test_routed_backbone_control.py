import copy
import unittest
from types import SimpleNamespace
from unittest.mock import patch
import torch
from scripts.vision import routed_backbone_control as arm

class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.model=torch.nn.Sequential(*[torch.nn.Sequential(torch.nn.Conv2d(3,3,1),torch.nn.BatchNorm2d(3)) for _ in range(12)])

def frozen_model():
    from ultralytics.models.yolo.detect import DetectionTrainer
    m=Model()
    for n,p in m.named_parameters():p.requires_grad_(not arm.frozen_name(n))
    DetectionTrainer._model_train(SimpleNamespace(model=m,freeze_layer_names=[f'model.{i}.' for i in arm.FROZEN]))
    return m

class BackboneTests(unittest.TestCase):
    def test_partition_boundary(self):
        self.assertTrue(arm.frozen_name('model.10.foo'))
        self.assertFalse(arm.frozen_name('model.11.foo'))
        self.assertFalse(arm.frozen_name('model.100.foo'))
    def test_actual_train_mode_keeps_frozen_bn(self):
        m=frozen_model();partition=arm.assert_policy(m);self.assertGreater(partition['trainable_parameters'],0)
        initial=arm.state_digest(m)
        with torch.no_grad():m.model(torch.randn(2,3,8,8))
        self.assertEqual(arm.state_digest(m),initial)
        self.assertTrue(m.model[11][1].training)
    def test_drift_and_grad_rejected(self):
        m=frozen_model();p=next(m.parameters());p.grad=torch.zeros_like(p)
        with self.assertRaises(ValueError):arm.assert_policy(m)
        p.grad=None;p.requires_grad_(True)
        with self.assertRaises(ValueError):arm.assert_policy(m)
        p.requires_grad_(False);m.model[0][1].train()
        with self.assertRaises(ValueError):arm.assert_policy(m)
    def test_ema_copy_does_not_change_head(self):
        m=frozen_model();ema=copy.deepcopy(m)
        with torch.no_grad():
            for p in ema.parameters():p.add_(1)
        before=arm.state_digest(ema,False)
        arm.copy_frozen_to_ema(m,ema)
        self.assertEqual(arm.state_digest(m),arm.state_digest(ema))
        self.assertEqual(before,arm.state_digest(ema,False))
    def test_digest_covers_buffers(self):
        m=frozen_model();initial=arm.state_digest(m)
        m.model[0][1].running_mean.add_(1)
        self.assertNotEqual(initial,arm.state_digest(m))
    def pair(self):
        old={k:[] for k in ('pool_rows','names','initialization','evaluation','environment','held_members')}
        old.update(training_config={'lr0':.00025},configuration={'lr':.00025})
        for f in ('schedules','exposures','windows','listings','contrast'):old[f]={k:[1,2,3] for k in arm.SOURCE_KEYS}
        p=copy.deepcopy(old)
        for f in ('schedules','exposures','windows','listings','contrast'):p[f]={k:old[f][o][:] for k,o in zip(arm.KEYS,arm.SOURCE_KEYS)}
        p['training_config']['freeze']=arm.FROZEN[:];p['configuration']['parameter_update_policy']=arm.VERSION;p['backbone_policy']=copy.deepcopy(arm.POLICY)
        return old,p
    def test_only_partition_changes(self):
        old,p=self.pair();arm.validate(old,p)
        for field in ('schedules','exposures','windows','listings','contrast'):
            bad=copy.deepcopy(p);bad[field][arm.KEYS[0]][0]=9
            with self.assertRaises(ValueError):arm.validate(old,bad)
        for field in ('pool_rows','held_members','environment'):
            bad=copy.deepcopy(p);bad[field]=['changed']
            with self.assertRaises(ValueError):arm.validate(old,bad)
        bad=copy.deepcopy(p);bad['training_config']['lr0']=.0005
        with self.assertRaises(ValueError):arm.validate(old,bad)
        bad=copy.deepcopy(p);bad['backbone_policy']['indices']=[0]
        with self.assertRaises(ValueError):arm.validate(old,bad)
    def test_previous_review_must_be_complete(self):
        rows=[{}, {'positive_decisions':[],'material_decisions':[],'negative_decisions':[]},{},{},
              {'status':'pending','integrity':{'integrity_passed':True,'pinned_files_verified':40}},{},{}]
        with patch.object(arm,'checked',side_effect=rows),patch.object(arm,'validate_positive'),patch.object(arm,'validate_review'):
            with self.assertRaises(ValueError):arm.review_gate()
    def test_completed_unit_requires_partition_evidence(self):
        with patch.object(arm,'checked',side_effect=FileNotFoundError('missing')):
            with self.assertRaises(FileNotFoundError):arm.verified_unit(arm.KEYS[0])
        rows=[{'optimizer_steps':480},{'steps':[{}]*479,'nonbackbone_changed':True,'terminal_frozen_half_equal':True},{},{}]
        with patch.object(arm,'checked',side_effect=rows):
            with self.assertRaises(ValueError):arm.verified_unit(arm.KEYS[0])

if __name__=='__main__':unittest.main()

import copy
import unittest
from unittest.mock import patch
from scripts.vision import brightness_lr_retention as m


class LearningRateControl(unittest.TestCase):
    def test_only_lr_changes(self):
        for seed in (7,17,27):
            old=m.original_overrides(seed); new=m.overrides(seed)
            self.assertEqual({k for k in old if old[k]!=new[k]}, {'lr0'})
            self.assertEqual(new['lr0'], .0005)
            self.assertEqual(m.original_overrides(seed)['lr0'], .001)

    def protocol(self):
        fields=('pool_rows','names','schedules','brightness_factors','listings','ledger','initialization','evaluation','acceptance_policy','retention','environment')
        return {**{k:[] for k in fields},'learning_rate':.0005,'reference_learning_rate':.001}

    def test_pair_rejects_any_non_lr_change(self):
        p=self.protocol(); m.check_pair(p,copy.deepcopy(p))
        for k in p:
            q=copy.deepcopy(p); q[k]='changed'
            with self.assertRaises(ValueError):m.check_pair(q,p)

    def test_curve_exact(self):
        curve=[{'train/box_loss':'1','lr/pg0':'.0005','lr/pg1':'.0005','lr/pg2':'.0005'} for _ in range(45)]
        m.check_curve(m.overrides(7),curve,7,.0005)
        for bad in (curve[:-1], [{**r,'lr/pg0':'.001'} for r in curve],
                    [{**r,'train/box_loss':'nan'} for r in curve],
                    [{k:v for k,v in r.items() if k!='lr/pg2'} for r in curve]):
            with self.assertRaises(ValueError):m.check_curve(m.overrides(7),bad,7,.0005)
        with self.assertRaises(ValueError):m.check_curve(m.original_overrides(7),curve,7,.0005)

    def test_preflight_rejects_sequence_pixels_and_training(self):
        key='brightness-450-7'; off='noaug-450-7';seq=['a']*2700
        log=[dict(position=i,member_id='a',gain=1.,before='same',after='same') for i in range(2700)]
        p={'identity':'p','schedules':{key:seq,off:seq},'brightness_factors':{key:[1.]*2700,off:[1.]*2700}}
        u=dict(protocol_identity='p',batches=450,draws=seq,logs={key:log,off:log},baseline_tensor_bytes_identical=True,complete_labels_identical=True,optimizer_created=False,backward_executed=False,validation_run=False)
        with patch.object(m,'verify'):
            m.validate_preflight(p,7,u,copy.deepcopy(u))
            for field,value in [('protocol_identity','wrong'),('batches',449),('draws',seq[:-1]),('logs',{}),('optimizer_created',True),('backward_executed',True),('validation_run',True)]:
                bad=copy.deepcopy(u);bad[field]=value
                with self.assertRaises(ValueError):m.validate_preflight(p,7,bad,u)


if __name__=='__main__':unittest.main()

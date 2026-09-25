import copy
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from scripts.vision import cool_light_control as control
from scripts.vision.cool_light_dataset import validate_review


class ControlledSwap(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        root = Path(self.tmp.name)
        image, label = root/'image', root/'label'
        image.write_bytes(b'image'); label.write_text('0 .5 .5 .2 .2\n')
        common = dict(image_path=str(image), image_sha256=control.prior.file_sha256(image),
                      label_path=str(label), label_sha256=control.prior.file_sha256(label),
                      subset='positive', class_instances={'transformer': 1}, lineage_id='pose')
        old = [dict(common, member_id=f'low-{i}', source_member_id=f'src-{i}', illumination='physical_low_neutral') for i in range(32)]
        self.new = [dict(common, member_id=f'cool-{i}', source_member_id=f'src-{i}') for i in range(32)]
        negative = dict(common, member_id='negative', subset='hard_negative', class_instances={})
        self.ref = dict(pool_rows=old+[negative], schedules={}, brightness_factors={}, training_config={})
        for name in ('initialization', 'evaluation', 'environment', 'acceptance_policy', 'names'):
            self.ref[name] = {}
        self.ref['held_members'] = ['held']
        for seed in (7,17,27):
            key = f'IL1000-{seed}'
            self.ref['schedules'][key] = [f'low-{i%32}' for i in range(300)]+['negative']*5700
            self.ref['brightness_factors'][key] = [1.]*6000
            self.ref['training_config'][key] = dict(epochs=100, lr0=.0005)
        self.p = copy.deepcopy(self.ref)
        self.p['pool_rows'] += self.new
        self.p.update(totals={}, ledger={})
        rows = {r['member_id']:r for r in self.p['pool_rows']}
        swap = control.substitution(self.ref, self.new)
        for key in control.KEYS:
            oldkey = 'IL1000-'+key.split('-')[-1]
            seq = [swap.get(x,x) for x in self.ref['schedules'][oldkey]]
            self.p['schedules'][key] = seq
            self.p['brightness_factors'][key] = self.ref['brightness_factors'][oldkey].copy()
            self.p['training_config'][key] = self.ref['training_config'][oldkey].copy()
            self.p['totals'][key] = control.counts(seq, rows)
            self.p['ledger'][key] = [dict(first_step=i//6+1, **control.counts(seq[i:i+300],rows)) for i in range(0,6000,300)]

    def test_valid(self): control.constraints(self.p,self.ref,self.new)
    def test_negative_position(self):
        self.p['schedules']['IC1000-7'][301] = 'cool-1'
        with self.assertRaises(ValueError): control.constraints(self.p,self.ref,self.new)
    def test_brightness(self):
        self.p['brightness_factors']['IC1000-7'][0] = .9
        with self.assertRaises(ValueError): control.constraints(self.p,self.ref,self.new)
    def test_learning_rate(self):
        self.p['training_config']['IC1000-7']['lr0'] = .00025
        with self.assertRaises(ValueError): control.constraints(self.p,self.ref,self.new)
    def test_label_drift(self):
        Path(self.new[0]['label_path']).write_text('1 .5 .5 .2 .2\n')
        with self.assertRaises(ValueError): control.constraints(self.p,self.ref,self.new)
    def test_duplicate_source(self):
        self.new[-1]['source_member_id'] = self.new[0]['source_member_id']
        with self.assertRaises(ValueError): control.substitution(self.ref,self.new)
    def test_missing_counterpart(self):
        with self.assertRaises(ValueError): control.substitution(self.ref,self.new[:-1])
    def test_ledger(self):
        self.p['ledger']['IC1000-7'][0]['classes']['transformer'] += 1
        with self.assertRaises(ValueError): control.constraints(self.p,self.ref,self.new)


class ExplicitReview(unittest.TestCase):
    def setUp(self):
        self.p = {'units':[{'unit_id':'L01'}]}
        self.e = dict(identity='e',events=[dict(event_id='L01:a',crop_sha256='c')])
        self.r = dict(full_frames_viewed=['L01'], decisions=[dict(event_id='L01:a', evidence_identity='e',crop_sha256='c',status='content_sufficient_for_bounded_research', reason='body visible',review_nature='AI辅助审核', training_admitted=False,promotable=False)])
    def validate(self):
        with patch('scripts.vision.cool_light_dataset.prior.verify'), patch('scripts.vision.cool_light_dataset.evidence',return_value=self.e):
            return validate_review(self.p,self.r)
    def test_explicit(self): self.validate()
    def test_missing(self):
        self.r['decisions']=[]
        with self.assertRaises(ValueError): self.validate()
    def test_duplicate(self):
        self.r['decisions']*=2
        with self.assertRaises(ValueError): self.validate()
    def test_stale(self):
        self.r['decisions'][0]['crop_sha256']='changed'
        with self.assertRaises(ValueError): self.validate()
    def test_unknown(self):
        self.r['decisions'][0]['status']='unknown'
        with self.assertRaises(ValueError): self.validate()


class EntryGate(unittest.TestCase):
    def setUp(self):
        from scripts.vision import train_cool_light as train
        self.train=train
        self.receipt=dict(status='ready_for_training_not_started',cells=list(control.KEYS),actual_draws_verified=18000,
            relevant_regressions_passed=True,exact_image_and_label_tensors_frozen=True,optimizer_created=False,
            backward_executed=False,training_admitted=False,promotable=False)
    def check(self):
        with patch.object(self.train.prior,'read',return_value=self.receipt),patch.object(self.train.prior,'verify'):
            return self.train.verify_ready()
    def test_valid(self): self.check()
    def test_status(self):
        self.receipt['status']='pilot_explicitly_reviewed_expand_frozen_32'
        with self.assertRaises(ValueError):self.check()
    def test_incomplete_draws(self):
        self.receipt['actual_draws_verified']=6000
        with self.assertRaises(ValueError):self.check()
    def test_missing_tests(self):
        self.receipt['relevant_regressions_passed']=False
        with self.assertRaises(ValueError):self.check()


if __name__=='__main__': unittest.main()

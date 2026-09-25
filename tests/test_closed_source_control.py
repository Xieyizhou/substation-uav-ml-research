import copy
import unittest
from unittest.mock import patch
from scripts.vision import freeze_closed_source_control as f
from scripts.vision import closed_source_training_quality as q
from scripts.vision.record_closed_exterior_review import validate


class ClosedSourceControlTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.p=f.protocol();cls.old,_,_=f.reference.contract('T-7')
        cls.review=f.prior.read(f.OUT/'quality-review.json')
        cls.evidence=[f.prior.read(ep) for _,_,ep in q.sources()]

    def checked(self,p):
        # Byte integrity is tested independently; avoid rehashing all images in every mutation test.
        hashes={r[t+'_path']:r[t+'_sha256'] for r in p['pool_rows'] for t in ('image','label')}
        with patch.object(f.prior,'file_sha256',side_effect=lambda path:hashes[str(path)]):f.check(p,self.old)

    def test_exact_design(self):
        self.checked(self.p)
        for seed in (7,17,27):
            self.assertEqual(self.p['totals'][f'E-{seed}']['classes'],self.p['totals'][f'M-{seed}']['classes'])
            rows={r['member_id']:r for r in self.p['pool_rows']}
            for family,original_count in (('E',30),('M',8)):
                seq=self.p['schedules'][f'{family}-{seed}']
                new=[rows[m] for m in seq if m.startswith('new-')]
                self.assertEqual(len(new),30)
                self.assertEqual(sum(r['variant']=='original' for r in new),original_count)

    def test_sequence_mutations_rejected(self):
        for kind in ('drop','reorder','brightness','negative','held'):
            p=copy.deepcopy(self.p);key='M-7';seq=p['schedules'][key]
            if kind=='drop':seq.pop()
            elif kind=='reorder':seq[0],seq[-1]=seq[-1],seq[0]
            elif kind=='brightness':p['brightness_factors'][key][0]+=.01
            elif kind=='negative':
                rows={r['member_id']:r for r in p['pool_rows']};i=next(i for i,m in enumerate(seq) if rows[m]['subset']=='hard_negative');seq[i]=seq[0]
            else:p['held_members'].append(seq[0])
            with self.subTest(kind=kind),self.assertRaises(ValueError):self.checked(p)

    def test_member_hash_rejected(self):
        p=copy.deepcopy(self.p);p['pool_rows'][0]['label_sha256']='wrong'
        with self.assertRaisesRegex(ValueError,'bytes changed'):f.check(p,self.old)

    def test_review_missing_duplicate_stale(self):
        ds=self.review['decisions'];validate(self.evidence,ds)
        for bad in (ds[:-1],ds+[ds[0]],[dict(ds[0],crop_sha256='stale')]+ds[1:]):
            with self.assertRaises(ValueError):validate(self.evidence,bad)

    def test_fresh_supplement_no_layout_or_asset_change(self):
        from scripts.vision import supplement_closed_transformer as s
        p=s.freeze()
        self.assertEqual(len(p['runs']),4)
        for run in p['runs']:
            plan=f.prior.read(run['plan_path']);old=f.prior.read(s.base.OUT/'plans/layout-B'/run['variant']/'plan.json')
            self.assertEqual(plan['files'],old['files'])
            self.assertEqual(plan['objects'],old['objects'])
            self.assertEqual(len(plan['calibration_views']),1)


if __name__=='__main__':unittest.main()

import copy
import unittest
from collections import Counter
from unittest.mock import patch,Mock
import subprocess
import tempfile
from pathlib import Path
from scripts.vision.freeze_compensated_double_dose import OUT,prior,source,checks,schedules
from scripts.vision import train_compensated_double_dose as entry


class DoseTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.old=prior.read(source.OUT/'protocol.json');cls.p=prior.read(OUT/'protocol.json')
        cls.counts=prior.read(OUT/'counts.json')

    def test_frozen_exact_sequences(self):
        checks(self.old,self.p)
        for seed in (7,17,27):
            a,b,positions=schedules(self.old,seed,self.counts['results'][str(seed)])
            self.assertEqual(a,self.p['schedules'][f'V-{seed}'])
            self.assertEqual(b,self.p['schedules'][f'VM-{seed}'])
            self.assertEqual(len(positions),54)
            self.assertEqual(sum('full_truth' in next(r for r in self.p['pool_rows'] if r['member_id']==m) for m in a),108)

    def test_invalid_protocol_rejected(self):
        lookup={r['member_id']:r for r in self.p['pool_rows']};key='V-7'
        for defect in ('missing','negative_position','brightness','config','new_dose','label','held'):
            p=copy.deepcopy(self.p)
            if defect=='missing':p['schedules'][key].pop()
            elif defect=='negative_position':
                seq=p['schedules'][key];i=next(i for i,m in enumerate(seq) if lookup[m]['subset']=='hard_negative')
                j=next(j for j,m in enumerate(seq) if lookup[m]['subset']=='hard_negative' and m!=seq[i]);seq[i],seq[j]=seq[j],seq[i]
            elif defect=='brightness':p['brightness_factors'][key][0]=.91
            elif defect=='config':p['training_config'][key]['lr0']=.001
            elif defect=='new_dose':p['schedules'][key][0]='C01-original'
            elif defect=='label':p['pool_rows'][-1]['full_truth']['objects'][0]['bbox_xyxy'][0]+=2
            elif defect=='held':p['schedules'][key][0]=p['held_members'][0]
            with self.subTest(defect=defect),self.assertRaises(ValueError):checks(self.old,p)

    def test_old_member_counts_never_increase_or_disappear(self):
        for key in entry.KEYS:
            a=Counter(self.old['schedules'][key]);b=Counter(self.p['schedules'][key])
            for r in self.p['pool_rows']:
                if 'full_truth' not in r:
                    self.assertLessEqual(b[r['member_id']],a[r['member_id']])
                    if a[r['member_id']]:self.assertGreaterEqual(b[r['member_id']],1)

    def test_worker_requires_explicit_flag(self):
        with patch.object(entry,'worker',side_effect=AssertionError('Must not run')):
            with self.assertRaises(SystemExit):entry.main(['--worker','V-7'])

    def test_cleanup_escalates_only_own_process(self):
        p=Mock();p.pid=123;p.poll.return_value=None;p.wait.side_effect=[subprocess.TimeoutExpired('own',10),0]
        with patch.object(entry.os,'killpg') as kill:
            entry.cleanup(p)
            self.assertEqual(kill.call_count,2)
            self.assertTrue(all(c.args[0]==123 for c in kill.call_args_list))

    def test_worker_attempt_limit(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp)
            for i in range(1,4):(root/'workers/V-7'/f'attempt-{i:03}').mkdir(parents=True)
            with patch.object(entry,'OUT',root),patch.object(entry,'KEYS',('V-7',)),patch.object(entry,'ready'),patch.object(entry.prior,'frozen'),patch.object(entry.prior,'file_sha256',return_value='identity'),patch.object(entry.subprocess,'Popen') as start:
                with self.assertRaisesRegex(ValueError,'Worker attempt cap'):entry.run()
                start.assert_not_called()
            self.assertEqual(len(list((root/'workers/V-7').glob('attempt-*'))),3)


if __name__=='__main__':unittest.main()

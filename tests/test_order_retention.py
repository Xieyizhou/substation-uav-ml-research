import copy
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch
from collections import Counter
import numpy as np
from scripts.vision.exposure_order_retention import permutation,validate_permutation,identity,file_sha256,read,OUT,PRIOR
from scripts.vision.order_retention_runtime import check_actual,validate_labels,overrides
from scripts.vision.review_exposure_order_retention import validate_decisions
from src.ml.artifacts import object_sha256

class OrderTests(unittest.TestCase):
    def fixture(self):
        seq=[f'm{i%90}' for i in range(2700)];vm={f'm{i}':('neutral_bridge' if i<10 else 'common') for i in range(90)}
        return seq,vm
    def test_permutation_reproducible_exact(self):
        seq,vm=self.fixture();b,ids=permutation(seq,vm,7)
        self.assertEqual((b,ids),permutation(seq,vm,7))
        self.assertNotEqual(ids,list(range(450)))
        self.assertEqual(Counter(tuple(x) for x in b),Counter(tuple(b[i]) for i in ids))
        self.assertEqual(Counter(seq),Counter(m for i in ids for m in b[i]))
    def test_uniform_cumulative_placement(self):
        seq,vm=self.fixture();b,ids=permutation(seq,vm,7);flags=[any(vm[m]=='neutral_bridge' for m in x) for x in b];n=sum(flags)
        for k in range(1,451):self.assertEqual(sum(flags[i] for i in ids[:k]),k*n//450)
    def test_missing_batch(self):
        seq,vm=self.fixture();b,ids=permutation(seq,vm,7)
        with self.assertRaises(ValueError):validate_permutation(b,ids[:-1])
    def test_duplicate_batch(self):
        seq,vm=self.fixture();b,ids=permutation(seq,vm,7);ids[0]=ids[1]
        with self.assertRaises(ValueError):validate_permutation(b,ids)
    def test_incomplete_batch(self):
        seq,vm=self.fixture()
        with self.assertRaises(ValueError):permutation(seq[:-1],vm,7)
    def test_unresolved_role(self):
        seq,vm=self.fixture();vm.pop('m0')
        with self.assertRaises(ValueError):permutation(seq,vm,7)
    def test_actual_order_wrong(self):
        seq,_=self.fixture();wrong=seq.copy();wrong[0],wrong[1]=wrong[1],wrong[0]
        with self.assertRaises(ValueError):check_actual({'schedules':{'k':seq}},'k',wrong)
    def test_actual_order_exact(self):
        seq,_=self.fixture();check_actual({'schedules':{'k':seq}},'k',seq.copy())
    def test_instance_identity_not_index(self):
        a={'annotation_id':'gazebo-truth-001-instance-0208-box-0000','class_name':'reactor','bbox_xyxy':[1,2,3,4]}
        b={**a,'annotation_id':'gazebo-truth-002-instance-0208-box-0009'}
        self.assertEqual(identity(a),identity(b))
        self.assertNotEqual(identity(a),identity({**b,'bbox_xyxy':[1,2,3,5]}))
    def test_unknown_instance_rejected(self):
        with self.assertRaises(ValueError):identity({'annotation_id':'class-mask'})
    def test_runtime_label_mutation(self):
        with tempfile.TemporaryDirectory() as tmp:
            p=Path(tmp)/'label.txt';p.write_text('3 0.5 0.5 0.2 0.2\n')
            rows={'m':dict(image_path='im',label_path=str(p),label_sha256=file_sha256(p))}
            ds=SimpleNamespace(labels=[dict(im_file='im',cls=np.array([[3]],dtype=np.float32),bboxes=np.array([[.5,.5,.2,.2]],dtype=np.float32))])
            validate_labels(ds,rows);ds.labels[0]['bboxes'][0,0]=.6
            with self.assertRaises(ValueError):validate_labels(ds,rows)
    def test_no_aug_and_budget(self):
        cfg=overrides(7);self.assertEqual(cfg['epochs']*10,450)
        for k in ('mosaic','mixup','cutmix','hsv_h','scale','multi_scale'):self.assertEqual(cfg[k],0)

class ReviewTests(unittest.TestCase):
    def fixture(self,tmp):
        ip=Path(tmp)/'image';ip.write_bytes(b'image');ep=Path(tmp)/'evidence';ep.write_bytes(b'evidence')
        e=dict(event_id='N01',image_path=str(ip),image_sha256=file_sha256(ip),evidence_path=str(ep),evidence_sha256=file_sha256(ep),prediction={'confidence':.5})
        d=dict(event_id='N01',source_event_identity=object_sha256(e),image_sha256=e['image_sha256'],evidence_sha256=e['evidence_sha256'],reason='observed',review_nature='AI辅助审核',reviewed_at='2026-09-09',status='diagnosed',content_category='柜体')
        return {'negative':[e],'reactors':[]},d
    def test_missing(self):
        with tempfile.TemporaryDirectory() as tmp:
            e,d=self.fixture(tmp)
            with self.assertRaises(ValueError):validate_decisions(e,[])
    def test_duplicate(self):
        with tempfile.TemporaryDirectory() as tmp:
            e,d=self.fixture(tmp)
            with self.assertRaises(ValueError):validate_decisions(e,[d,d])
    def test_stale_image(self):
        with tempfile.TemporaryDirectory() as tmp:
            e,d=self.fixture(tmp);Path(e['negative'][0]['image_path']).write_bytes(b'changed')
            with self.assertRaises(ValueError):validate_decisions(e,[d])
    def test_stale_prediction(self):
        with tempfile.TemporaryDirectory() as tmp:
            e,d=self.fixture(tmp);e['negative'][0]['prediction']['confidence']=.6
            with self.assertRaises(ValueError):validate_decisions(e,[d])
    def test_stale_evidence(self):
        with tempfile.TemporaryDirectory() as tmp:
            e,d=self.fixture(tmp);Path(e['negative'][0]['evidence_path']).write_bytes(b'changed')
            with self.assertRaises(ValueError):validate_decisions(e,[d])
    def test_unresolved_blocks(self):
        with tempfile.TemporaryDirectory() as tmp:
            e,d=self.fixture(tmp);d['content_category']='无法确认'
            with self.assertRaises(ValueError):validate_decisions(e,[d])
    def test_explicit_valid(self):
        with tempfile.TemporaryDirectory() as tmp:
            e,d=self.fixture(tmp);validate_decisions(e,[d])

class EntryTests(unittest.TestCase):
    def test_default_never_trains(self):
        import scripts.vision.train_order_retention as entry
        with patch.object(entry,'validate_ready',return_value={}),patch.object(entry,'isolated') as launch,patch('sys.argv',['entry']):
            entry.main();launch.assert_not_called()
    def test_worker_requires_explicit_train(self):
        import scripts.vision.train_order_retention as entry
        with patch('sys.argv',['entry','--worker','staged-450-7']):
            with self.assertRaises(SystemExit):entry.main()
    def test_missing_ready(self):
        import scripts.vision.preflight_order_retention as entry
        with tempfile.TemporaryDirectory() as tmp,patch.object(entry,'OUT',Path(tmp)):
            with self.assertRaises(FileNotFoundError):entry.validate_ready()
    def test_stale_ready_input(self):
        import scripts.vision.preflight_order_retention as entry
        from scripts.vision.exposure_protocol import save
        with tempfile.TemporaryDirectory() as tmp,patch.object(entry,'OUT',Path(tmp)):
            p=Path(tmp)/'protocol.json';p.write_text('{}')
            save(Path(tmp)/'ready.json',dict(status='ready_for_training_not_started',cells=list(entry.KEYS),inputs={str(p):file_sha256(p)}))
            p.write_text('{"changed":true}')
            with self.assertRaises(ValueError):entry.validate_ready()
    def test_cancel_cleans_worker(self):
        import scripts.vision.train_order_retention as entry
        fake=SimpleNamespace(pid=123,returncode=None)
        fake.poll=lambda:None
        calls=[]
        def wait(timeout):
            calls.append(timeout)
            if len(calls)==1:raise KeyboardInterrupt()
            fake.returncode=-15
        fake.wait=wait
        with tempfile.TemporaryDirectory() as tmp,patch.object(entry,'OUT',Path(tmp)),patch.object(entry.subprocess,'Popen',return_value=fake),patch.object(entry.os,'killpg') as kill:
            with self.assertRaises(KeyboardInterrupt):entry.isolated('staged-450-7')
            kill.assert_called_once();self.assertEqual(calls,[21600,10])
            failure=next(Path(tmp).glob('workers/*/*/failure.json'));self.assertEqual(read(failure)['status'],'failed')
    def test_worker_retries_bounded(self):
        import scripts.vision.train_order_retention as entry
        with tempfile.TemporaryDirectory() as tmp,patch.object(entry,'OUT',Path(tmp)),patch.object(entry.subprocess,'Popen') as launch:
            root=Path(tmp)/'workers/staged-450-7'
            for i in range(3):(root/f'attempt-{i+1:03}').mkdir(parents=True)
            with self.assertRaises(ValueError):entry.isolated('staged-450-7')
            launch.assert_not_called()

if __name__=='__main__':unittest.main()

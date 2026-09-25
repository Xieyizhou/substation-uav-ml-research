from copy import deepcopy
from pathlib import Path
import ast
import inspect
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch, Mock
from PIL import Image
from scripts.vision.check_structure_fit_sources import equal_rgb, label_correspondence, validate_mapping, strict_review, strict_unit
from scripts.vision.structure_fit import STRUCTURES, file_sha256, frozen, read, scoring
from scripts.vision.run_structure_fit import cleanup, worker, validate_unit
from scripts.vision import recover_structure_fit as recovery


class StructureIntegrityTests(unittest.TestCase):
    def test_rgb_lossless_and_any_pixel_or_size_change(self):
        with tempfile.TemporaryDirectory() as folder:
            a,b=Path(folder)/'a.ppm',Path(folder)/'b.png'
            im=Image.new('RGB',(16,12),(120,40,20));im.save(a);im.save(b)
            self.assertNotEqual(file_sha256(a),file_sha256(b));equal_rgb(a,b)
            im.putpixel((0,0),(121,40,20));im.save(b)
            with self.assertRaises(ValueError):equal_rgb(a,b)
            Image.new('RGB',(12,16),(120,40,20)).save(b)
            with self.assertRaises(ValueError):equal_rgb(a,b)

    def test_full_labels_no_count_identity_or_coordinate_fallback(self):
        a=dict(class_name='reactor',bbox_xyxy=[0,0,10,10])
        b=dict(class_name='reactor',bbox_xyxy=[20,0,30,10])
        self.assertEqual(label_correspondence([a,b],[b,a]),[a,b])
        for actual in ([a],[a,a],[a,dict(b,class_name='transformer')],[a,dict(b,bbox_xyxy=[21,0,31,10])]):
            with self.assertRaises(ValueError):label_correspondence([a,b],actual)

    def test_instance_mapping_collision_and_unparseable(self):
        a={'object_id':'reactor_north','category':'reactor'}
        validate_mapping({'208':a})
        for m in ({'208':a,'209':a},{'unknown':a},{'208':{'object_id':'reactor_north'}}):
            with self.assertRaises((KeyError,ValueError)):validate_mapping(m)

    def test_strict_review_invalid_roi_and_unknown_retained(self):
        with tempfile.TemporaryDirectory() as folder:
            ip=Path(folder)/'a.png';Image.new('RGB',(10,10)).save(ip)
            e=dict(event_id='N001',group='negative',image_path=str(ip),image_sha256=file_sha256(ip),evidence_path=str(ip),evidence_sha256=file_sha256(ip),source_event_identity='src')
            d={k:e[k] for k in ('event_id','image_sha256','evidence_sha256','source_event_identity')}
            d.update(review_nature='AI辅助审核',reason='explicit',reviewed_at='2026-09-09',pixel_visibility_certified=False,structures={k:dict(state='unknown',reason='not resolved') for k in STRUCTURES})
            strict_review({'events':[e]},[d])
            invalid=deepcopy(d);invalid['structures']['front_panel']=dict(state='present',reason='roi',roi_xyxy=[0,0,11,10])
            with self.assertRaises(ValueError):strict_review({'events':[e]},[invalid])
            invalid=deepcopy(d);invalid['pixel_visibility_certified']=True
            with self.assertRaises(ValueError):strict_review({'events':[e]},[invalid])

    def test_multiple_instance_exposures_zero_seen_and_matching_recomputed(self):
        t=[dict(class_name='reactor',bbox_xyxy=[0,0,10,10],annotation_id='a'),dict(class_name='reactor',bbox_xyxy=[20,0,30,10],annotation_id='b')]
        source=dict(member_id='one',image_sha256='rgb')
        p={'models':{'model':{'draws':['one','one','one']}},'rows':[source],'development':[]}
        row=dict(member_id='one',image_sha256='rgb',actual_exposures=3,**scoring(t,[],[]))
        result=dict(pool=[row],development=[],optimizer_created=False,backward_executed=False,training_validation_executed=False)
        with patch('scripts.vision.run_structure_fit.validate_unit'),patch('scripts.vision.check_structure_fit_sources.truth_for',return_value=t):
            strict_unit(result,'model',p)
            self.assertEqual(len(t)*row['actual_exposures'],6)
            bad=deepcopy(result);bad['pool'][0]['actual_exposures']=0
            with self.assertRaises(ValueError):strict_unit(bad,'model',p)
            bad=deepcopy(result);bad['pool'][0]['misses']=[]
            with self.assertRaises(ValueError):strict_unit(bad,'model',p)
            p['models']['model']['draws']=[];result['pool'][0]['actual_exposures']=0
            strict_unit(result,'model',p)

    def test_prediction_competes_for_two_truths(self):
        t=[dict(class_name='reactor',bbox_xyxy=[0,0,10,10]),dict(class_name='reactor',bbox_xyxy=[1,0,11,10])]
        pred=[dict(**t[0],confidence=.8)]
        r=scoring(t,pred,pred)
        self.assertEqual(len(r['matches']),1);self.assertTrue(r['misses'][0]['formal_matching_competition'])
        self.assertNotEqual(r['misses'][0]['reason'],'low_confidence_same_class')

    def test_publish_complete_only_and_hash_valid_resume(self):
        with tempfile.TemporaryDirectory() as folder,patch.object(recovery,'OUT',Path(folder)):
            out=Path(folder);(out/'inference').mkdir();attempt=out/'attempt-001';attempt.mkdir()
            frozen(out/'publication-repair.json',dict(status='repair',inputs={}))
            p={'identity':'protocol','rows':[{'member_id':str(i)} for i in range(236)],'development':[{'view_id':str(i),'variant':'x'} for i in range(96)]}
            source=out/'input';source.write_bytes(b'original')
            payload=dict(status='complete',model='test',protocol_identity='protocol',pool=p['rows'],development=p['development'],inputs={str(source):file_sha256(source)})
            frozen(attempt/'result.json',payload)
            recovery.publish('test',attempt,p)
            first=file_sha256(out/'inference/test.json');recovery.publish('test',attempt,p)
            self.assertEqual(first,file_sha256(out/'inference/test.json'))
            source.write_bytes(b'changed')
            with self.assertRaises(ValueError):recovery.publish('test',attempt,p)

    def test_incomplete_unit_not_reused(self):
        with tempfile.TemporaryDirectory() as folder:
            r=frozen(Path(folder)/'unit.json',dict(status='complete',model='x',protocol_identity='p',pool=[],development=[],inputs={}))
            with self.assertRaises(ValueError):validate_unit(r,'x',{'identity':'p','rows':[],'development':[]})

    def test_retry_limit_and_cleanup_after_failure(self):
        with tempfile.TemporaryDirectory() as folder,patch.object(recovery,'OUT',Path(folder)):
            out=Path(folder);(out/'inference').mkdir();frozen(out/'protocol.json',dict(models={'bad':{}},inputs={}))
            proc=Mock();proc.wait.return_value=1
            with patch.object(sys,'argv',['recovery','--infer']),patch.object(recovery.subprocess,'Popen',return_value=proc) as spawn,patch.object(recovery,'cleanup') as clean:
                with self.assertRaisesRegex(ValueError,'Attempt limit'):recovery.main()
                self.assertEqual(spawn.call_count,3);self.assertEqual(clean.call_count,3)
            self.assertEqual(len(list((out/'inference/bad').glob('attempt-*/failure.json'))),3)
            self.assertFalse((out/'inference.lock').exists())

    def test_cancellation_cleanup_and_lock_release(self):
        with tempfile.TemporaryDirectory() as folder,patch.object(recovery,'OUT',Path(folder)):
            out=Path(folder);(out/'inference').mkdir();frozen(out/'protocol.json',dict(models={'bad':{}},inputs={}))
            proc=Mock();proc.wait.side_effect=KeyboardInterrupt()
            with patch.object(sys,'argv',['recovery','--infer']),patch.object(recovery.subprocess,'Popen',return_value=proc),patch.object(recovery,'cleanup') as clean:
                with self.assertRaises(KeyboardInterrupt):recovery.main()
                clean.assert_called_once_with(proc)
            self.assertFalse((out/'inference.lock').exists())

    def test_real_worker_process_cleanup(self):
        proc=subprocess.Popen([sys.executable,'-c','import time; time.sleep(20)'],start_new_session=True)
        try:cleanup(proc);self.assertIsNotNone(proc.poll())
        finally:
            if proc.poll() is None:proc.kill();proc.wait()

    def test_inference_worker_contains_no_training_calls(self):
        tree=ast.parse(inspect.getsource(worker))
        names=[n.func.attr for n in ast.walk(tree) if isinstance(n,ast.Call) and isinstance(n.func,ast.Attribute)]
        self.assertFalse(set(names)&{'train','val','backward','step','zero_grad'})


if __name__=='__main__':unittest.main()

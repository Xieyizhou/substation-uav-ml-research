import copy
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch
import torch
from scripts.vision import reviewed_saturation_control as m


def test_factors_are_balanced_deterministic_and_seed_specific():
    assert m.factors(7)==m.factors(7)
    assert m.factors(7)!=m.factors(17)
    assert m.Counter(m.factors(7))=={0.:960,.5:960,1.:960}


def test_color_only_endpoints_and_no_input_mutation():
    x=torch.tensor([[[[255,0]],[[0,255]],[[0,0]]]],dtype=torch.uint8)
    saved=x.clone()
    assert torch.equal(m.desaturate(x,[1.]),x)
    gray=m.desaturate(x,[0.])
    assert torch.equal(gray[:,0],gray[:,1]) and torch.equal(gray[:,1],gray[:,2])
    assert torch.equal(x,saved) and gray.shape==x.shape
    assert gray[0,0,0].tolist()==[76,150]


def test_invalid_factors_fail():
    for values in ([.7],[],[0.,1.]):
        with unittest.TestCase().assertRaises(ValueError):m.desaturate(torch.zeros((1,3,2,2),dtype=torch.uint8),values)


def test_non_rgb_or_float_rejected():
    with unittest.TestCase().assertRaises(ValueError):m.desaturate(torch.zeros((1,3,2,2)),[0.])
    with unittest.TestCase().assertRaises(ValueError):m.desaturate(torch.zeros((1,1,2,2),dtype=torch.uint8),[0.])


def protocol_pair():
    old=dict(pool_rows=[],initialization={},schedules={},exposures={})
    p=dict(pool_rows=[],initialization={},schedules={},exposures={},saturation={})
    for seed,key in zip(m.SEEDS,m.KEYS):
        old['schedules'][f'reviewed-interleaved-480-{seed}']=['a','b']*1440
        old['exposures'][f'reviewed-interleaved-480-{seed}']={'draws':2880}
        p['schedules'][key]=['a','b']*1440;p['exposures'][key]={'draws':2880};p['saturation'][key]=m.factors(seed)
    return p,old


def test_schedule_and_factor_drift_rejected():
    p,old=protocol_pair();m.validate_protocol(p,old)
    q=copy.deepcopy(p);q['schedules'][m.KEYS[0]][0]='c'
    with unittest.TestCase().assertRaises(ValueError):m.validate_protocol(q,old)
    q=copy.deepcopy(p);q['saturation'][m.KEYS[0]][0]=.3
    with unittest.TestCase().assertRaises(ValueError):m.validate_protocol(q,old)


def test_explicit_chain_calls_evaluation_after_training():
    calls=[]
    with tempfile.TemporaryDirectory() as directory, patch.object(m,'OUT',Path(directory)),patch.object(m,'preflight'),patch.object(m,'checked',return_value={}),patch.object(m,'file_sha256',return_value='test'),patch.object(m,'write_record'),patch.object(m,'parallel',side_effect=lambda flag,keys,folder:calls.append((flag,tuple(keys)))),patch.object(m,'summarize',side_effect=lambda:calls.append(('summary',))):
        m.train()
    assert calls==[('--worker',m.KEYS),('--eval-worker',m.KEYS),('summary',)]


def test_actual_wrapper_preserves_labels_and_rejects_stale_tensor():
    from types import SimpleNamespace
    batch=dict(img=torch.full((6,3,2,2),42,dtype=torch.uint8),cls=torch.tensor([[1.]]),bboxes=torch.tensor([[.5,.5,.2,.2]]),batch_idx=torch.tensor([0]),im_file=['a']*6)
    p={'saturation':{'unit':[0.,.5,1.,0.,.5,1.]}}
    with tempfile.TemporaryDirectory() as directory,patch.object(m,'OUT',Path(directory)),patch.object(m.source,'loader',return_value=(object(),[copy.deepcopy(batch)])):
        m.OBSERVED['unit']=[]
        _,wrapper=m.loader(p,'unit',SimpleNamespace(epoch=0));out=list(wrapper)[0]
        assert torch.equal(out['cls'],batch['cls']) and torch.equal(out['bboxes'],batch['bboxes'])
        expected=copy.deepcopy(m.OBSERVED['unit'])
        root=Path(directory)/'actual-preflight';root.mkdir();(root/'unit.json').touch()
        expected[0]['augmented_tensor_sha256']='stale'
        with patch.object(m,'checked',return_value={'tensor_records':expected}):
            _,wrapper=m.loader(p,'unit',SimpleNamespace(epoch=0))
            with unittest.TestCase().assertRaises(ValueError):list(wrapper)


def load_tests(loader, tests, pattern):
    return unittest.TestSuite(unittest.FunctionTestCase(v) for k,v in globals().items() if k.startswith('test_') and callable(v))

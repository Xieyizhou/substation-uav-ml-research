"""Trace permutation equivalence from real images through one training update."""
import sys,copy,json
from pathlib import Path
import torch,yaml
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT))
from ultralytics.models.yolo.detect import DetectionTrainer
from scripts.vision.run_order_diagnosis import OUT as ORDER,REFERENCE,read,save,file_sha256,verify_tree
OUT=ORDER/'permutation-root-cause-v1'

def flat(value,prefix=''):
    if isinstance(value,torch.Tensor):return {prefix:value}
    if isinstance(value,dict):return {k2:v2 for k,v in value.items() for k2,v2 in flat(v,prefix+'/'+str(k)).items()}
    if isinstance(value,(tuple,list)):return {k2:v2 for k,v in enumerate(value) for k2,v2 in flat(v,prefix+'/'+str(k)).items()}
    return {}

def difference(a,b):
    a=a.detach().cpu().double();b=b.detach().cpu().double();d=(a-b).abs()
    return dict(max_abs=float(d.max()) if d.numel() else 0.,mean_abs=float(d.mean()) if d.numel() else 0.,changed=int((d!=0).sum()),count=d.numel())

def aligned(a,b,inverse):
    aa,bb=flat(a),flat(b)
    return {k:difference(t,bb[k][inverse] if bb[k].ndim and bb[k].shape[0]==len(inverse) else bb[k]) for k,t in aa.items()}

def setup():
    args=yaml.safe_load((REFERENCE/'F-100-7/attempt-001/args.yaml').read_text())
    args.update(project=str(OUT),name='setup',exist_ok=True,plots=False,save=False)
    args.pop('save_dir',None)
    t=DetectionTrainer(overrides=args);t._setup_train();t.model.train()
    return t

def get_batches(t,offset=0):
    p=read(REFERENCE/'protocol.json');w=read(ORDER/'protocol.json')
    rows={r['member_id']:r for r in p['pool_rows']};dataset=t.train_loader.dataset;mapping={path:i for i,path in enumerate(dataset.im_files)}
    ids=[p['schedules']['F-100-7'][offset:offset+6],w['schedules']['W-100-7'][offset:offset+6]]
    result=[]
    for seq in ids:
        result.append(t.preprocess_batch(dataset.collate_fn([dataset[mapping[rows[mid]['image_path']]] for mid in seq])))
    available=list(range(6));inverse=[]
    for mid in ids[0]:
        pos=next(i for i in available if ids[1][i]==mid);available.remove(pos);inverse.append(pos)
    return result,ids,inverse

def run_pair(base,batch_pair,inverse,mode='train'):
    models=[copy.deepcopy(base),copy.deepcopy(base)];outputs=[];hooks=[];captures=[{},{}];losses=[];assignments=[]
    for index,(model,batch) in enumerate(zip(models,batch_pair)):
        model.train(mode!='eval')
        if mode=='frozen_bn':
            for module in model.modules():
                if isinstance(module,torch.nn.BatchNorm2d):module.eval()
        handles=[]
        for name,module in model.named_modules():
            if isinstance(module,(torch.nn.Conv2d,torch.nn.BatchNorm2d)):
                def hook(m,args,out,name=name,index=index):captures[index][name]=out.detach().clone()
                handles.append(module.register_forward_hook(hook))
        output=model(batch['img']);outputs.append(output)
        for handle in handles:handle.remove()
        criterion=model.init_criterion();assignment=[]
        handle=criterion.assigner.register_forward_hook(lambda m,args,out:assignment.append(out))
        loss,items=criterion(output,batch);handle.remove();assignments.append(assignment[0]);losses.append(dict(total=loss.detach().tolist(),items={k:float(v) for k,v in items.items()}))
        loss.sum().backward()
    gradient={name:difference(p.grad,dict(models[1].named_parameters())[name].grad) for name,p in models[0].named_parameters() if p.grad is not None}
    layer_differences={name:difference(value,captures[1][name][inverse]) for name,value in captures[0].items()}
    state={name:difference(value,models[1].state_dict()[name]) for name,value in models[0].state_dict().items()}
    return dict(mode=mode,layers=layer_differences,outputs=aligned(outputs[0],outputs[1],inverse),losses=losses,
        assignments=aligned(assignments[0],assignments[1],inverse),gradients=gradient,buffers=state)

def main():
    verify_tree(ORDER/'report-receipt.json');t=setup();pair,ids,inverse=get_batches(t)
    checks=dict(images=difference(pair[0]['img'],pair[1]['img'][inverse]),targets=[])
    for i,j in enumerate(inverse):
        checks['targets'].append({k:difference(pair[0][k][pair[0]['batch_idx']==i],pair[1][k][pair[1]['batch_idx']==j]) for k in ('cls','bboxes')})
    results={mode:run_pair(t.model,pair,inverse,mode) for mode in ('eval','train','frozen_bn')}
    paths=[Path(__file__),ORDER/'report-receipt.json']
    import inspect
    from ultralytics.utils.loss import v8DetectionLoss
    from ultralytics.utils.tal import TaskAlignedAssigner
    paths += [Path(inspect.getfile(cls)) for cls in (DetectionTrainer,v8DetectionLoss,TaskAlignedAssigner)]
    r=save(OUT/'single-batch.json',dict(status='complete',ids=ids,inverse=inverse,input_checks=checks,results=results,torch_threads=torch.get_num_threads(),
        inputs={str(p):file_sha256(p) for p in paths}))
    print('INPUTS',checks)
    for mode,v in results.items():
        print(mode,'first_layer_differences',[(k,d['max_abs']) for k,d in v['layers'].items() if d['changed']][:5],
            'loss',v['losses'],'assignments',v['assignments'],'gradient_max',max(d['max_abs'] for d in v['gradients'].values()))

if __name__=='__main__':main()

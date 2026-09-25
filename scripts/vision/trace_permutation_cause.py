"""Paired real-data AdamW replay, first divergence and canonical-order control."""
import sys,copy,json
from pathlib import Path
import torch
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT))
import scripts.vision.probe_batch_permutation as probe
from scripts.vision.canonical_batch_order import canonicalize
from scripts.vision.run_order_diagnosis import OUT as ORDER,REFERENCE,read,save,file_sha256,verify_tree
OUT=ORDER/'permutation-root-cause-v1'

def batches(t,step,canonical=False):
    p=read(REFERENCE/'protocol.json');w=read(ORDER/'protocol.json');rows={r['member_id']:r for r in p['pool_rows']}
    d=t.train_loader.dataset;mapping={path:i for i,path in enumerate(d.im_files)}
    ids=[p['schedules']['F-100-7'][step*6:step*6+6],w['schedules']['W-100-7'][step*6:step*6+6]]
    if canonical:ids=[canonicalize(x) for x in ids]
    values=[t.preprocess_batch(d.collate_fn([d[mapping[rows[m]['image_path']]] for m in seq])) for seq in ids]
    left=list(range(6));inverse=[]
    for mid in ids[0]:
        j=next(j for j in left if ids[1][j]==mid);left.remove(j);inverse.append(j)
    return values,ids,inverse

def replay(t,steps,canonical=False):
    pairs=[copy.deepcopy((t.model,t.optimizer)),copy.deepcopy((t.model,t.optimizer))]
    log=[]
    for step in range(steps):
        data,ids,inverse=batches(t,step,canonical);losses=[];assign=[];outputs=[];score_sums=[]
        for (model,opt),batch in zip(pairs,data):
            model.train();opt.zero_grad();pred=model(batch['img']);outputs.append({k:v.detach() for k,v in pred.items() if isinstance(v,torch.Tensor)})
            if not hasattr(model,'criterion'):model.criterion=model.init_criterion()
            assigned=[];handle=model.criterion.assigner.register_forward_hook(lambda m,args,out:assigned.append(out))
            loss,items=model.criterion(pred,batch);handle.remove();loss.sum().backward();losses.append(loss.detach().tolist());assign.append(assigned[0])
            scores=assigned[0][2];score_sums.append(dict(fp32=float(scores.sum()),fp64=float(scores.double().sum())))
        gradients={n:probe.difference(v.grad,dict(pairs[1][0].named_parameters())[n].grad) for n,v in pairs[0][0].named_parameters() if v.grad is not None}
        gradmax=max(d['max_abs'] for d in gradients.values())
        for model,opt in pairs:
            torch.nn.utils.clip_grad_norm_(model.parameters(),10.);opt.step();opt.zero_grad()
        differences={n:probe.difference(v,pairs[1][0].state_dict()[n]) for n,v in pairs[0][0].state_dict().items()}
        parameter_max=max(differences[n]['max_abs'] for n,_ in pairs[0][0].named_parameters())
        r=dict(step=step+1,ids=ids,losses=losses,target_score_sums=score_sums,
            gradient_max=gradmax,parameter_max=parameter_max,state_max=max(d['max_abs'] for d in differences.values()),
            raw_output_differences=probe.aligned(outputs[0],outputs[1],inverse),assignment_differences=probe.aligned(assign[0],assign[1],inverse))
        log.append(r)
        print('TRACE',canonical,step+1,'grad',gradmax,'parameter',parameter_max,'fg_changed',r['assignment_differences']['/3']['changed'],flush=True)
        del pred,assigned,assign,outputs,gradients,differences
    return log

def main():
    verify_tree(OUT/'single-batch.json');probe.OUT=OUT/'trace-setup';t=probe.setup()
    raw=replay(t,20,False);fixed=replay(t,20,True)
    if any(r['gradient_max'] or r['state_max'] for r in fixed):raise ValueError('Canonical order did not restore exact equality')
    save(OUT/'trace.json',dict(status='complete',raw=raw,canonical=fixed,intervention='Stable member-ID sorting before fetch/collate, preserving batch composition and order.',
        inputs={str(p):file_sha256(p) for p in (Path(__file__),ROOT/'scripts/vision/canonical_batch_order.py',OUT/'single-batch.json')}))

if __name__=='__main__':main()

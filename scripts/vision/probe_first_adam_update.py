"""Quantify the first AdamW update amplification at affected tensor elements."""
import sys,copy
from pathlib import Path
import torch
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT))
import scripts.vision.probe_batch_permutation as probe
from scripts.vision.run_order_diagnosis import OUT as ORDER,read,save,file_sha256
OUT=ORDER/'permutation-root-cause-v1'

def main():
    probe.OUT=OUT/'first-step-setup';t=probe.setup();b,ids,inverse=probe.get_batches(t)
    pairs=[copy.deepcopy((t.model,t.optimizer)),copy.deepcopy((t.model,t.optimizer))];grads=[];norms=[]
    for (m,o),batch in zip(pairs,b):
        m.train();loss,_=m(batch);loss.sum().backward()
        grads.append({n:p.grad.detach().clone() for n,p in m.named_parameters() if p.grad is not None})
        norms.append(float(torch.nn.utils.clip_grad_norm_(m.parameters(),10.)));o.step()
    changes=[]
    for name,a in pairs[0][0].named_parameters():
        if name not in grads[0]:continue
        z=dict(pairs[1][0].named_parameters())[name];d=(a-z).detach().abs();index=int(d.flatten().argmax())
        if float(d.max())==0:continue
        changes.append(dict(name=name,max_update_difference=float(d.max()),flat_index=index,
            gradient_F=float(grads[0][name].flatten()[index]),gradient_W=float(grads[1][name].flatten()[index]),
            value_F=float(a.detach().flatten()[index]),value_W=float(z.detach().flatten()[index]),
            initial_value=float(dict(t.model.named_parameters())[name].detach().flatten()[index])))
    changes.sort(key=lambda r:r['max_update_difference'],reverse=True)
    save(OUT/'first-adam-update.json',dict(status='complete',unclipped_gradient_norms=norms,clip_scale=[min(1,10/(n+1e-6)) for n in norms],
        adam_epsilon=t.optimizer.defaults['eps'],top_differences=changes,
        inputs={str(p):file_sha256(p) for p in (Path(__file__),OUT/'single-batch.json')}))
    print('NORMS',norms,'EPS',t.optimizer.defaults['eps']);print('TOP',changes[:5])

if __name__=='__main__':main()

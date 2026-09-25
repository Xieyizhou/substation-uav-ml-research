"""Deterministic material-only, no-crop scale transform of actual loader tensors."""
import hashlib
import torch
from scripts.vision.material_routed_contrast_control import MATERIAL,UNCHANGED

VERSION='material-half-positions-center-scale075-v1'

def factors(rows,sequence,seed):
    by={r['member_id']:r for r in rows}
    if len(by)!=len(rows) or any(r['variant'] not in MATERIAL|UNCHANGED for r in rows):raise ValueError('Unresolved member or variant')
    if any(m not in by for m in sequence):raise ValueError('Unknown member')
    positions=[i for i,m in enumerate(sequence) if by[m]['variant'] in MATERIAL]
    positions.sort(key=lambda i:hashlib.sha256(f'{VERSION}:{seed}:{i}'.encode()).digest())
    selected=set(positions[:len(positions)//2])
    return [.75 if i in selected else 1. for i in range(len(sequence))]

def transform(batch,values):
    images=batch['img'];boxes=batch['bboxes'];indices=batch['batch_idx']
    if images.dtype!=torch.uint8 or images.ndim!=4 or tuple(images.shape[1:])!=(3,640,640):raise ValueError('Expected uint8 RGB640 batch')
    if len(values)!=len(images) or any(v not in (1.,.75) for v in values):raise ValueError('Unfrozen scale')
    if boxes.ndim!=2 or boxes.shape[1]!=4 or len(boxes)!=len(indices) or len(boxes)!=len(batch['cls']):raise ValueError('Malformed full labels')
    if not torch.isfinite(boxes).all() or not torch.isfinite(indices).all():raise ValueError('Nonfinite label')
    if len(indices) and ((indices!=indices.round()).any() or (indices<0).any() or (indices>=len(images)).any()):raise ValueError('Wrong batch index')
    if len(boxes) and ((boxes[:,2:]<=0).any() or (boxes[:,:2]-boxes[:,2:]/2 < -1e-6).any() or (boxes[:,:2]+boxes[:,2:]/2 > 1+1e-6).any()):raise ValueError('Out-of-image box')
    result=dict(batch);out=images.clone();labels=boxes.clone()
    for i,value in enumerate(values):
        if value==1.:continue
        small=torch.nn.functional.interpolate(images[i:i+1].float(),size=(480,480),mode='bilinear',align_corners=False,antialias=True).round().clamp(0,255).to(torch.uint8)
        out[i].fill_(114);out[i,:,80:560,80:560]=small[0]
        mask=indices==i
        labels[mask,:2]=boxes[mask,:2]*.75+.125
        labels[mask,2:]=boxes[mask,2:]*.75
    result['img']=out;result['bboxes']=labels
    return result

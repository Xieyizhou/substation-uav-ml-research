"""Bounded material-routed nonlinear intensity control, not physical relighting."""
import torch
VERSION='material-routed-gamma-after-contrast-v1'

def factors(contrast):
    mapping={.75:.8,1.:1.,1.25:1.25}
    if any(x not in mapping for x in contrast):raise ValueError('Unknown frozen contrast')
    return [mapping[x] for x in contrast]

def transform(images,values):
    if images.dtype!=torch.uint8 or images.ndim!=4 or images.shape[1]!=3 or len(values)!=len(images):raise ValueError('Input contract')
    if any(x not in (.8,1.,1.25) for x in values):raise ValueError('Unfrozen gamma')
    out=images.clone()
    for i,g in enumerate(values):
        if g!=1.:out[i]=((images[i].float()/255.).pow(g)*255.).round().clamp(0,255).to(torch.uint8)
    return out

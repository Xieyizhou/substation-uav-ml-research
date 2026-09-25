"""Frozen whole-frame shrink after brightness; complete labels are retained."""
import copy
import numpy as np
from PIL import Image
from scripts.vision import closed_budget_runtime as base
from scripts.vision.brightness_transfer_runtime import pixel_hash

class Scale:
    def __init__(self,p,key,log):self.p=p;self.key=key;self.log=log;self.n=0
    def __call__(self,labels):
        i=self.n;self.n+=1
        if i>=len(self.p['scale_factors'][self.key]) or len(self.log)!=i+1:raise ValueError('Scale sequence drift')
        factor=self.p['scale_factors'][self.key][i];im=labels['img'];h,w=im.shape[:2]
        nw,nh=round(w*factor),round(h*factor);ox,oy=(w-nw)//2,(h-nh)//2
        if factor not in (1.,.75,.5):raise ValueError('Unfrozen scale')
        before=pixel_hash(im);inst=copy.deepcopy(labels['instances']);inst.convert_bbox('xyxy');inst.denormalize(w,h)
        if factor!=1:
            scaled=np.array(Image.fromarray(im).resize((nw,nh),Image.Resampling.BILINEAR))
            out=np.full_like(im,114);out[oy:oy+nh,ox:ox+nw]=scaled
            inst.scale(nw/w,nh/h);inst.add_padding(ox,oy);labels['instances']=inst;labels['img']=out
        if len(inst)!=len(labels['cls']):raise ValueError('Full labels lost')
        self.log[-1].update(scale=factor,scale_before=before,scale_after=pixel_hash(labels['img']),
            scale_shape=[h,w],scale_after_boxes=inst.bboxes.tolist(),scale_classes=labels['cls'].reshape(-1).tolist())
        return labels

def make_dataset(p,key,log):
    d=base.make_dataset(p,key,log);d.transforms.insert(1,Scale(p,key,log));return d

make_loader=base.make_loader
def check(p,key,actual,log):
    base.check(p,key,actual,log)
    for i,r in enumerate(log):
        if r['scale']!=p['scale_factors'][key][i]:raise ValueError('Scale log drift')
        if r['scale']==1 and r['scale_before']!=r['scale_after']:raise ValueError('Identity scale changed pixels')

def verify_supervision(batch,logs):
    for i,r in enumerate(logs):
        mask=batch['batch_idx'].cpu().numpy().reshape(-1)==i
        cls=batch['cls'].cpu().numpy().reshape(-1)[mask];boxes=batch['bboxes'].cpu().numpy()[mask]
        if not np.array_equal(cls,np.asarray(r['scale_classes'])):raise ValueError('Full class instances changed')
        b=np.asarray(r['scale_after_boxes'],dtype=np.float64).reshape(-1,4);h,w=r['scale_shape']
        ratio=min(640/h,640/w);dw=round((640-round(w*ratio))/2-.1);dh=round((640-round(h*ratio))/2-.1)
        b*=ratio;b+=np.array([dw,dh,dw,dh]);expected=np.empty_like(b)
        expected[:,:2]=(b[:,:2]+b[:,2:])/2/640;expected[:,2:]=(b[:,2:]-b[:,:2])/640
        if boxes.shape!=expected.shape or not np.allclose(boxes,expected,rtol=0,atol=2e-6):raise ValueError('Complete label geometry changed')

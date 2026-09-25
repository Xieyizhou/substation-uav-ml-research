"""Deterministic V-channel gamma after existing brightness; geometry unchanged."""
import cv2
import numpy as np
from scripts.vision import closed_budget_runtime as base
from scripts.vision.brightness_transfer_runtime import pixel_hash

def transform(image,gamma):
    if gamma not in (.8,1.,1.2):raise ValueError('Unfrozen gamma')
    if image.dtype!=np.uint8 or image.ndim!=3 or image.shape[2]!=3:raise ValueError('Expected uint8 BGR')
    if gamma==1.:return image.copy()
    hsv=cv2.cvtColor(image,cv2.COLOR_BGR2HSV)
    lut=np.rint(255*(np.arange(256,dtype=np.float64)/255)**gamma).astype(np.uint8)
    hsv[:,:,2]=cv2.LUT(hsv[:,:,2],lut)
    return cv2.cvtColor(hsv,cv2.COLOR_HSV2BGR)

class Gamma:
    def __init__(self,p,key,log):self.p=p;self.key=key;self.log=log;self.n=0;self.members={r['image_path']:r['member_id'] for r in p['pool_rows']}
    def __call__(self,labels):
        if self.n>=len(self.p['schedules'][self.key]):raise ValueError('Extra gamma exposure')
        member=self.members[labels['im_file']]
        if member!=self.p['schedules'][self.key][self.n]:raise ValueError('Gamma member order drift')
        gamma=self.p['gamma_factors'][self.key][self.n//6];before=labels['img'];after=transform(before,gamma)
        self.log.append(dict(position=self.n,member_id=member,gamma=gamma,before=pixel_hash(before),after=pixel_hash(after)))
        labels['img']=after;self.n+=1;return labels

def make_dataset(p,key,brightness_log,gamma_log):
    dataset=base.make_dataset(p,key,brightness_log);dataset.transforms.insert(1,Gamma(p,key,gamma_log));return dataset

make_loader=base.make_loader

def check(p,key,actual,brightness_log,gamma_log):
    base.check(p,key,actual,brightness_log)
    if len(gamma_log)!=len(actual):raise ValueError('Incomplete gamma ledger')
    for i,(g,b) in enumerate(zip(gamma_log,brightness_log,strict=True)):
        if (g['position'],g['member_id'],g['gamma'])!=(i,actual[i],p['gamma_factors'][key][i//6]) or g['before']!=b['after']:raise ValueError('Gamma pipeline mismatch')
        if g['gamma']==1 and g['before']!=g['after']:raise ValueError('Identity gamma changed bytes')

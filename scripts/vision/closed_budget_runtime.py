"""Variable-budget frozen loader. Historical 2700-exposure runtime is untouched."""
from scripts.vision import brightness_transfer_runtime as brightness
from scripts.vision import order_retention_runtime as base

class Brightness(brightness.Brightness):
    def __call__(self,labels):
        if self.n>=len(self.p['schedules'][self.key]):raise ValueError('Extra exposure')
        member=self.idx[labels['im_file']]
        if member!=self.p['schedules'][self.key][self.n]:raise ValueError('Position mismatch')
        gain=self.p['brightness_factors'][self.key][self.n]
        before=labels['img'];after=brightness.brighten(before,gain)
        self.log.append(dict(position=self.n,member_id=member,gain=gain,before=brightness.pixel_hash(before),after=brightness.pixel_hash(after)))
        labels['img']=after;self.n+=1;return labels

def make_dataset(p,key,log):
    dataset=base.make_dataset(p,key)
    for t in dataset.transforms.transforms:
        if type(t).__name__=='Albumentations' and t.transform is not None:raise ValueError('Unplanned augmentation')
    dataset.transforms.insert(0,Brightness(p,key,log));return dataset

make_loader=base.make_loader

def check(p,key,actual,log):
    seq=p['schedules'][key]
    if actual!=seq or len(log)!=len(seq):raise ValueError('Incomplete exposure')
    for i,r in enumerate(log):
        if (r['position'],r['member_id'],r['gain'])!=(i,seq[i],p['brightness_factors'][key][i]):raise ValueError('Actual brightness mismatch')
        if r['gain']==1 and r['before']!=r['after']:raise ValueError('Identity brightness mismatch')

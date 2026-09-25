import unittest
from types import SimpleNamespace
import torch
from scripts.vision.frozen_multiscale_runtime import preprocess

class MultiscaleRuntimeTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(2)
        self.owner=SimpleNamespace(device=torch.device('cpu'),args=SimpleNamespace(multi_scale=0),stride=32)
        self.batch=dict(img=torch.zeros((6,3,640,640),dtype=torch.uint8),cls=torch.tensor([[0.],[3.]]),
            bboxes=torch.tensor([[.5,.5,.2,.3],[.25,.3,.1,.2]]),batch_idx=torch.tensor([0.,5.]))

    def test_shapes_and_supervision(self):
        for size in (320,640,960):
            out=preprocess(self.owner,self.batch,size)
            self.assertEqual(tuple(out['img'].shape),(6,3,size,size))
            self.assertEqual(out['img'].dtype,torch.float32)
            self.assertTrue(torch.equal(out['bboxes'],self.batch['bboxes']))
            self.assertTrue(torch.equal(out['cls'],self.batch['cls']))
            self.assertTrue(torch.equal(out['batch_idx'],self.batch['batch_idx']))
            self.assertTrue(torch.allclose(out['bboxes']*size/size,self.batch['bboxes']))
        self.assertEqual(self.batch['img'].dtype,torch.uint8)

    def test_identity_normalization_and_spatial_interpolation(self):
        self.batch['img'][:,:,:,320:]=255
        fixed=preprocess(self.owner,self.batch,640)
        self.assertTrue(torch.equal(fixed['img'],self.batch['img'].float()/255))
        small=preprocess(self.owner,self.batch,320)
        self.assertTrue(torch.all(small['img'][:,:,:,:159]==0))
        self.assertTrue(torch.all(small['img'][:,:,:,161:]==1))

    def test_unfrozen_or_double_resize_rejected(self):
        with self.assertRaises(ValueError):preprocess(self.owner,self.batch,800)
        self.owner.args.multi_scale=.5
        with self.assertRaises(ValueError):preprocess(self.owner,self.batch,640)

    def test_wrong_input_dtype_rejected(self):
        self.batch['img']=self.batch['img'].float()
        with self.assertRaises(ValueError):preprocess(self.owner,self.batch,640)

    def test_recorded_shape_and_identity_gate(self):
        from scripts.vision.finalize_multiscale_loader import paired_checks
        import copy
        b=dict(members=['a']*6,size=640,source640_sha256='source',full_supervision={'cls':'x'},effective_shape=[6,3,640,640],effective_tensor_sha256='same')
        r=dict(arms={'fixed':{'batches':[b]},'multiscale':{'batches':[copy.deepcopy(b)]}})
        paired_checks(r)
        for field,value in [('effective_shape',[6,3,320,320]),('effective_tensor_sha256','different'),('full_supervision',{'cls':'different'})]:
            x=copy.deepcopy(r);x['arms']['multiscale']['batches'][0][field]=value
            with self.assertRaises(ValueError):paired_checks(x)

if __name__=='__main__':unittest.main()

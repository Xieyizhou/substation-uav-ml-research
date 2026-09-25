import copy
import unittest
from scripts.vision.material_routed_contrast_control import route,validate,KEYS,PRIOR_KEYS,VERSION

class RoutingTests(unittest.TestCase):
    def test_transform_is_per_image_and_identity_positions_exact(self):
        import torch
        from scripts.vision.contrast_transfer_control import transform
        x=torch.arange(6*3*4*4).remainder(256).byte().reshape(6,3,4,4)
        global_values=[.75,1.25,.75,1.25,1.,.75];routed=[.75,1.,.75,1.,1.,1.]
        all_images,_=transform(x,global_values);new,_=transform(x,routed)
        self.assertTrue(torch.equal(new[[0,2]],all_images[[0,2]]))
        self.assertTrue(torch.equal(new[[1,3,4,5]],x[[1,3,4,5]]))
    def test_positions_and_material_factors_preserved(self):
        rows=[{'member_id':'a','variant':'gray035'},{'member_id':'b','variant':'original'},{'member_id':'c','variant':'physical-lighting'}]
        seq=['a','b','c','a'];f=[.75,1.25,.75,1.25]
        self.assertEqual(route(rows,seq,f),[.75,1.,1.,1.25]);self.assertEqual(f,[.75,1.25,.75,1.25])
    def test_unresolved_and_duplicates_fail(self):
        rows=[{'member_id':'a','variant':'gray035'}]
        for rs,seq,values in [(rows*2,['a'],[1.]),([{'member_id':'a','variant':'unknown'}],['a'],[1.]),(rows,['missing'],[1.]),(rows,['a'],[]),(rows,['a'],[2.])]:
            with self.assertRaises(ValueError):route(rs,seq,values)
    def test_only_routing_changes(self):
        old={f:[] for f in ('names','initialization','evaluation','environment','held_members','training_config')}
        old['pool_rows']=[{'member_id':'a','variant':'neutral'},{'member_id':'b','variant':'original'}];old['configuration']={'lr':.00025,'augmentation':'global'}
        old['schedules']={k:['a','b'] for k in PRIOR_KEYS};old['contrast']={k:[.75,1.25] for k in PRIOR_KEYS}
        for f in ('exposures','windows','listings'):old[f]={k:[1] for k in PRIOR_KEYS}
        new=copy.deepcopy(old);new['configuration']['augmentation']=VERSION
        for f in ('schedules','exposures','windows','listings'):new[f]={k:old[f][o] for k,o in zip(KEYS,PRIOR_KEYS)}
        new['contrast']={k:[.75,1.] for k in KEYS};validate(old,new)
        for f in ('schedules','contrast'):
            bad=copy.deepcopy(new);bad[f][KEYS[0]]=[]
            with self.assertRaises(ValueError):validate(old,bad)
        bad=copy.deepcopy(new);bad['pool_rows'][0]['variant']='warm'
        with self.assertRaises(ValueError):validate(old,bad)

if __name__=='__main__':unittest.main()

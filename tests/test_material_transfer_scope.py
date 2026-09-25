import copy
import unittest
import xml.etree.ElementTree as ET
from scripts.vision.audit_material_transfer_scope import resolve_truth
from scripts.vision.freeze_condition_transfer_probe import derive
from scripts.vision.build_material_view_world_drafts import check_only_materials,signature


class ScopeTests(unittest.TestCase):
    def test_mapping_requires_unique_resolved_instance(self):
        t={'objects':[dict(annotation_id='truth-instance-0007-box-0',class_name='reactor',bbox_xyxy=[0,0,2,2])]}
        m={'7':dict(object_id='r',category='reactor')}
        self.assertEqual(resolve_truth(t,m)[0]['object_id'],'r')
        for mm in ({},{'7':dict(object_id='r',category='switchgear')},{**m,'8':m['7']}):
            with self.assertRaises(ValueError):resolve_truth(t,mm)
        x=copy.deepcopy(t);x['objects']*=2
        with self.assertRaises(ValueError):resolve_truth(x,m)

    def fixture(self):
        return ET.fromstring('<sdf><world>'+''.join(
            '<model name="'+name+'"><link name="link">'+''.join(
                '<visual name="'+v+'"><geometry><box><size>1 1 1</size></box></geometry><material><ambient>0.1 0.2 0.3 1</ambient><diffuse>0.1 0.2 0.3 1</diffuse></material></visual>'
                for v in visuals)+'</link></model>'
            for name,visuals in [('target',['body']),('other',['body','front_panel'])])+'</world></sdf>')

    def test_scope_isolation_and_panel_noop(self):
        b=self.fixture();names={'target','other'}
        target,changes=derive(b,'target',names,'gray_target_body');self.assertEqual(len(changes),2)
        full,_=derive(b,'target',names,'gray_target_full');self.assertEqual(signature(target),signature(full))
        allbody,c=derive(b,'target',names,'gray_all_body');self.assertEqual(len(c),4)
        allfull,c=derive(b,'target',names,'gray_all_full');self.assertEqual(len(c),6)
        allfull.find('.//size').text='2 2 2'
        with self.assertRaises(ValueError):check_only_materials(b,allfull,names)

    def test_unknown_target_refused(self):
        with self.assertRaises(ValueError):derive(self.fixture(),'absent',{'target','other'},'gray_target_body')


if __name__=='__main__':unittest.main()

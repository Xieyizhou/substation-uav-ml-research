import copy
import unittest
import xml.etree.ElementTree as ET
from scripts.vision.record_order_fit_remaining_review import validate
from scripts.vision.trace_order_fit_legacy_worlds import resolve_labels
from scripts.vision.test_body_material_applicability import model_mapping
from scripts.vision.extend_order_fit_quality_isolation import extend


class ReviewTests(unittest.TestCase):
    def setUp(self):
        self.t = dict(annotation_id='current', class_name='switchgear',bbox_xyxy=[0,0,10,10])
        self.e = dict(review_id='U01',member_id='m',image_sha256='im',label_sha256='lb',page_sha256='pg',truth=[self.t])
        self.d = dict(self.e,label_observations=[dict(truth_index=0,status='limited_content',reason='Only side visible')],
            full_frame_observation='Whole image reviewed; identity still pending',reviewed_at='2026-09-15',
            review_nature='AI辅助审核',pixel_visibility_certified=False,training_eligible=False)

    def test_complete_observation_is_not_admission(self):
        validate(dict(events=[self.e]),[self.d])
        d=copy.deepcopy(self.d);d['training_eligible']=True
        with self.assertRaises(ValueError): validate(dict(events=[self.e]),[d])

    def test_missing_duplicate_or_stale_decisions(self):
        for ds in ([],[self.d,self.d]):
            with self.assertRaises(ValueError): validate(dict(events=[self.e]),ds)
        for key in ('image_sha256','label_sha256','page_sha256'):
            d=copy.deepcopy(self.d);d[key]='changed'
            with self.assertRaises(ValueError): validate(dict(events=[self.e]),[d])

    def test_complete_labels_required(self):
        d=copy.deepcopy(self.d);d['label_observations']=[]
        with self.assertRaises(ValueError): validate(dict(events=[self.e]),[d])

    def test_actual_annotation_maps_not_planned_target(self):
        original=dict(self.t,annotation_id='gazebo-instance-0042-box-0')
        r=resolve_labels([self.t],[original],{'42':'nonplanned_device','99':'planned_device'})
        self.assertEqual(r[0]['object_id'],'nonplanned_device')
        with self.assertRaises(ValueError):resolve_labels([self.t],[original],{'99':'planned_device'})

    def test_class_or_coordinates_change_rejected(self):
        original=dict(self.t,annotation_id='gazebo-instance-0042-box-0')
        for delta in (dict(class_name='transformer'),dict(bbox_xyxy=[0,0,12,10])):
            with self.assertRaises(ValueError): resolve_labels([self.t],[dict(original,**delta)],{'42':'x'})

    def test_world_instance_collision(self):
        model='<model name="{}"><link><visual><plugin name="gz::sim::systems::Label"><label>42</label></plugin></visual></link></model>'
        tree=ET.ElementTree(ET.fromstring('<sdf><world>'+model.format('a')+model.format('b')+'</world></sdf>'))
        with self.assertRaises(ValueError):model_mapping(tree)

    def test_new_risk_propagates_without_restoring_old_holds(self):
        members=[dict(member_id=k,lineage_id=lineage) for k,lineage in [('a','same'),('b','same'),('c','other')]]
        previous=[dict(member_id=m['member_id'],status='quarantined' if m['member_id']=='c' else 'requires_complete_quality_and_role_validation',reasons=[]) for m in members]
        ds=[dict(member_id='a',label_observations=[dict(status='insufficient_content')])]
        rows=extend(members,previous,ds)
        self.assertTrue(all(x['status']=='quarantined' and not x['training_eligible'] for x in rows))
        ds[0]['label_observations'][0]['status']='limited_content'
        rows=extend(members,previous,ds)
        self.assertEqual(rows[0]['status'],'requires_complete_quality_and_role_validation')
        self.assertEqual(rows[2]['status'],'quarantined')


if __name__=='__main__':unittest.main()

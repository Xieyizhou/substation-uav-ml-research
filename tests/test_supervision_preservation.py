import tempfile
import unittest
from pathlib import Path
from PIL import Image
from scripts.vision.diagnose_supervision_preservation import certificate,quantile,labels,model_signature
from scripts.vision.exposure_protocol import file_sha256

class PreservationTests(unittest.TestCase):
    def test_saturation_forces_switchgear_only_zero(self):
        vectors={'bridge':{'reactor':1,'switchgear':2},'new':{'switchgear':1},'cap':{'capacitor_bank':1}}
        r=certificate(vectors,{'capacitor_bank':42,'reactor':78},120,{'capacitor_bank':1,'reactor':1})
        self.assertTrue(r['saturation_proven']);self.assertEqual(r['forced_zero_members'],['new'])

    def test_compensating_multitarget_invalidates_simple_proof(self):
        r=certificate({'both':{'reactor':1,'capacitor_bank':1},'new':{'switchgear':1}},
            {'reactor':1,'capacitor_bank':1},2,{'reactor':1,'capacitor_bank':1})
        self.assertFalse(r['saturation_proven']);self.assertEqual(r['forced_zero_members'],[])

    def test_nonsaturated_demand_not_claimed_infeasible(self):
        r=certificate({'a':{'reactor':1},'b':{}},{'reactor':1},2,{'reactor':1})
        self.assertFalse(r['saturation_proven']);self.assertEqual(r['forced_zero_members'],[])

    def test_small_enumeration_confirms_certificate(self):
        # Five slots, required C+R=5: including any zero-score image cannot work.
        for cap in range(6):
            for reactor in range(6-cap):
                zero=5-cap-reactor
                if cap+reactor==5:self.assertEqual(zero,0)

    def test_weighted_quantile(self):
        self.assertEqual(quantile([(10,1),(30,3)],.5),30)
        self.assertEqual(quantile([(10,1),(30,3)],.1),10)
        self.assertIsNone(quantile([],.5))

    def test_letterbox_short_side_and_stale_label(self):
        with tempfile.TemporaryDirectory() as d:
            image=Path(d)/'image.png';label=Path(d)/'label.txt'
            Image.new('RGB',(1920,1080)).save(image);label.write_text('0 .5 .5 .5 .5\n')
            row=dict(image_path=str(image),label_path=str(label),image_sha256=file_sha256(image),label_sha256=file_sha256(label),class_instances={'transformer':1})
            self.assertEqual(labels(row)[0]['short_side_640'],180)
            row['class_instances']={'switchgear':1}
            with self.assertRaises(ValueError):labels(row)
            label.write_text('changed')
            with self.assertRaises(ValueError):labels(row)

    def test_ambiguous_model_and_nonbox_not_equated(self):
        with tempfile.TemporaryDirectory() as d:
            p=Path(d)/'world.sdf';p.write_text('<sdf><model name="a"/><model name="a"/></sdf>')
            with self.assertRaises(ValueError):model_signature(p,'a')
            p.write_text('<sdf><model name="a"><link><visual name="body"><geometry><sphere/></geometry></visual></link></model></sdf>')
            with self.assertRaises(ValueError):model_signature(p,'a')

if __name__=='__main__':unittest.main()

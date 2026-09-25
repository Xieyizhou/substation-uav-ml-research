import copy
import tempfile
import unittest
from pathlib import Path
from PIL import Image
from scripts.vision.condition_transfer_design import same_source,unique,summarize_targets,development


class ConditionCensus(unittest.TestCase):
    def test_lossless_and_full_label_identity(self):
        with tempfile.TemporaryDirectory() as tmp:
            p=Path(tmp);Image.new('RGB',(10,10),'red').save(p/'a.png');Image.new('RGB',(10,10),'red').save(p/'b.bmp')
            # Test fixtures only; production never edits labels.
            (p/'a.txt').write_text('1 0.5 0.5 0.2 0.2\n');(p/'b.txt').write_text('1 .5 .5 .2 .2\n')
            self.assertTrue(same_source(str(p/'a.png'),str(p/'a.txt'),str(p/'b.bmp'),str(p/'b.txt')))
            Image.new('RGB',(10,10),'blue').save(p/'c.png')
            self.assertFalse(same_source(str(p/'a.png'),str(p/'a.txt'),str(p/'c.png'),str(p/'b.txt')))
            (p/'c.txt').write_text('1 .5 .5 .3 .2\n')
            self.assertFalse(same_source(str(p/'a.png'),str(p/'a.txt'),str(p/'b.bmp'),str(p/'c.txt')))

    def test_duplicate_identity(self):
        with self.assertRaises(ValueError):unique([{'id':'x'},{'id':'x'}],lambda x:x['id'])

    def test_counts_are_instances_not_independent_images(self):
        r=dict(truth={'class_name':'switchgear'},member_id='m',lineage_id='g',panel_condition='visible_low_contrast',linkage_status='linked',actual_exposures={'7':24,'17':24,'27':26})
        x=summarize_targets([r,r,r])['switchgear']['panel']['visible_low_contrast']
        self.assertEqual(x['members'],1);self.assertEqual(x['frame_label_events'],3);self.assertEqual(x['instance_exposures']['7'],72)

    def test_seed_repeats_and_identity_conflict(self):
        row=dict(view_id='v',variant='material',pair_id='p',image_sha256='image',truth=[dict(annotation_id='id',class_name='switchgear',bbox_xyxy=[1,2,3,4])],matches=[],misses=[dict(truth_index=0,reason='wrong_class')])
        records=[{'rows':[copy.deepcopy(row)]} for _ in range(3)]
        targets,counts=development(records)
        self.assertEqual(len(targets),1);self.assertEqual(counts['material']['switchgear']['persistent_misses'],1)
        self.assertEqual(counts['material']['switchgear']['miss_event_reasons']['wrong_class'],3)
        records[1]['rows'][0]['truth'][0]['annotation_id']='other'
        with self.assertRaises(ValueError):development(records)


if __name__=='__main__':unittest.main()

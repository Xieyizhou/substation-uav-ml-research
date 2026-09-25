import copy
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch
from PIL import Image
from scripts.vision import diagnose_gray_body_transfer as d


class DiagnosisTests(unittest.TestCase):
    def test_actual_pixels_and_full_truth(self):
        with tempfile.TemporaryDirectory() as tmp:
            a,b=Path(tmp)/'a.png',Path(tmp)/'b.bmp'
            Image.new('RGB',(4,4),'gray').save(a)
            Image.open(a).save(b)
            t=dict(class_name='reactor',bbox_xyxy=[0,0,4,4])
            row=dict(member_id='m',target='device',truth=[dict(t,object_id='device')],image_path=str(a))
            member=dict(member_id='m',full_truth=dict(objects=[t]),image_path=str(b))
            d.check_gray(row,member)
            broken=copy.deepcopy(member);broken['full_truth']['objects'][0]['bbox_xyxy'][0]=1
            with self.assertRaises(ValueError):d.check_gray(row,broken)
            im=Image.open(b);im.putpixel((0,0),(0,0,0));im.save(b)
            with self.assertRaises(ValueError):d.check_gray(row,member)

    def test_default_no_inference(self):
        with patch('sys.argv',['diagnose']),patch.object(d,'freeze',return_value={'members':[]}),patch.object(d,'infer') as run:
            d.main()
        run.assert_not_called()

    def test_no_training(self):
        with self.assertRaises(RuntimeError):d.forbidden()

    def test_incomplete_output(self):
        with patch.object(d.prior,'verify'):
            with self.assertRaises(ValueError):
                d.validate(dict(status='complete',model='G-7',protocol_identity='p',rows=[]),'G-7',dict(identity='p',members=[{}]))

    def test_target_resolves_by_identity_not_index(self):
        from scripts.vision.summarize_gray_fit_transfer import target_index
        self.assertEqual(target_index({'target':'b'},{'truth':[{'object_id':'a'},{'object_id':'b'}]}),1)
        for truth in ([{'object_id':'a'}],[{'object_id':'b'},{'object_id':'b'}]):
            with self.assertRaises(ValueError):target_index({'target':'b'},{'truth':truth})


if __name__=='__main__':unittest.main()

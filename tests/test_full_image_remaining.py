import unittest
from scripts.vision.capture_full_image_remaining import translate,MATRIX,VARIANTS
from scripts.vision.finalize_full_image_remaining import validate_matrix

class RemainingTests(unittest.TestCase):
    def test_exact_palette(self):
        for variant in VARIANTS:
            name,m=translate(variant,MATRIX)
            color,light=variant.split('_',1)
            self.assertEqual(name,'steel_'+light)
            self.assertEqual(m['materials']['steel'],MATRIX['materials'][color])
            self.assertEqual(m['lights'],MATRIX['lights'])
    def test_unknown_rejected(self):
        with self.assertRaises(ValueError):translate('grey_original',MATRIX)
    def test_matrix_missing_and_duplicate(self):
        originals=[dict(frame_id=f'F{i:02}') for i in range(1,9)]
        variants=[dict(original_frame_id=f'F{i:02}',variant=v,pair_status='aligned',pair_bbox_max_delta=0) for i in range(1,9) for v in ('steel_original','steel_warm_dim',*VARIANTS)]
        validate_matrix(originals,variants)
        with self.assertRaises(ValueError):validate_matrix(originals,variants[:-1])
        with self.assertRaises(ValueError):validate_matrix(originals,variants[:-1]+[variants[0]])
        variants[0]['pair_bbox_max_delta']=2
        with self.assertRaises(ValueError):validate_matrix(originals,variants)

import unittest
from scripts.vision.run_physical_lighting_capture import certify_pair

class CaptureTests(unittest.TestCase):
    def test_original_exact_required(self):
        with self.assertRaises(ValueError):certify_pair('original',False,[],[],0,[],True)
    def test_lighting_rgb_difference_expected(self):
        certify_pair('physical-lighting',False,[],[],0,[],True)
    def test_mask_membership_change_blocks(self):
        with self.assertRaises(ValueError):certify_pair('physical-lighting',False,[],[],0,[],False)
    def test_full_label_conflicts_block(self):
        for added,lost,delta,unboxed in [(['1'],[],0,[]),([],['2'],0,[]),([],[],1.001,[]),([],[],0,['3'])]:
            with self.assertRaises(ValueError):certify_pair('original',True,added,lost,delta,unboxed,True)

if __name__=='__main__':unittest.main()

import unittest
from scripts.vision.verify_visibility_repair_design import full_labels_equal

class FullLabelTests(unittest.TestCase):
    def test_same_size_shifted_box_rejected(self):
        with self.assertRaises(ValueError):full_labels_equal('1 .4 .5 .2 .3\n','1 .6 .5 .2 .3\n')

    def test_missing_box_rejected(self):
        with self.assertRaises(ValueError):full_labels_equal('1 .4 .5 .2 .3\n','')

    def test_exact_labels(self):
        full_labels_equal('1 .4 .5 .2 .3\n','1 .4 .5 .2 .3\n')

if __name__=='__main__':unittest.main()

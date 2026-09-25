"""Integration regressions against the frozen, real replacement protocol."""
import copy
import unittest
from scripts.vision.freeze_gray_body_control import freeze,checks,source,prior


class GrayControlTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.p=freeze();cls.old=prior.read(source.OUT/'protocol.json')

    def test_actual_frozen_counts(self):
        checks(self.p,self.old)
        self.assertEqual(len(self.p['pool_rows']),289)
        self.assertEqual([len(x) for x in self.p['replacement_positions'].values()],[30,30,30])

    def test_changed_brightness_rejected(self):
        p=copy.deepcopy(self.p);p['brightness_factors']['G-7'][0]=0.812345
        with self.assertRaises(ValueError):checks(p,self.old)

    def test_unplanned_position_rejected(self):
        p=copy.deepcopy(self.p);i=next(i for i in range(2700) if i not in p['replacement_positions']['7'])
        p['schedules']['G-7'][i]='S01-gray_target_body'
        with self.assertRaises(ValueError):checks(p,self.old)

    def test_missing_exposure_rejected(self):
        p=copy.deepcopy(self.p);p['schedules']['G-17'].pop()
        with self.assertRaises(ValueError):checks(p,self.old)


if __name__=='__main__':unittest.main()

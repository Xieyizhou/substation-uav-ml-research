import unittest
from pathlib import Path

class LowerRateTests(unittest.TestCase):
    def test_trainer_only_lr_differs(self):
        root=Path(__file__).resolve().parents[1]
        original=(root/'scripts/vision/run_exposure_diagnosis.py').read_text().split('def main():')[0].strip()
        current=(root/'scripts/vision/train_lower_rate_control.py').read_text().strip()
        self.assertEqual(current.replace('lr0=.0003,','lr0=.001,'),original)
        self.assertEqual(current.count('lr0=.0003,'),1)

if __name__=='__main__':unittest.main()

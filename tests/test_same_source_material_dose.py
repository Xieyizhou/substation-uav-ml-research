import unittest
from collections import Counter
from unittest.mock import patch
from scripts.vision.freeze_same_source_material_dose import allocate

class DoseTests(unittest.TestCase):
    def test_exact_and_reproducible(self):
        slots=[(i,f's{i}') for i in range(11)]+[(i,'repeat') for i in range(11,15)]
        for seed in (7,17,27):
            a=allocate(slots,seed);self.assertEqual(a,allocate(list(reversed(slots)),seed))
            self.assertEqual(sorted(Counter(a.values()).values()),[5,5,5])
            self.assertEqual(sorted(Counter(a[i] for i in range(11,15)).values()),[1,1,2])
    def test_infeasible_no_relaxation(self):
        with self.assertRaises(ValueError):allocate([(i,'x') for i in range(14)],7)
    def test_default_no_training(self):
        from scripts.vision import train_same_source_material_dose as t
        with patch.object(t,'ready'),patch.object(t,'run') as run,patch.object(t,'worker') as worker:
            t.main([]);run.assert_not_called();worker.assert_not_called()
            with self.assertRaises(SystemExit):t.main(['--worker','D-7'])

if __name__=='__main__':unittest.main()

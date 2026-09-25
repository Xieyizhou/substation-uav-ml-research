import unittest
from scripts.vision.benchmark_reviewed_hold_device import check_prefix,KEY,CELLS

class DeviceBenchmarkTests(unittest.TestCase):
    def test_exact_prefix(self):
        p={'schedules':{KEY:list(range(2700))}}
        check_prefix(p,list(range(180)),list(range(180)),list(range(2700)))
        for draws,log in [(list(range(179)),list(range(180))),
                          (list(reversed(range(180))),list(range(180))),
                          (list(range(180)),[0]*180)]:
            with self.assertRaises(ValueError):check_prefix(p,draws,log,list(range(2700)))

    def test_bounded_counterbalanced_cells(self):
        self.assertEqual(CELLS,('cpu-1','mps-1','mps-2','cpu-2'))

if __name__=='__main__':unittest.main()

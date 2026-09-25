import unittest
import torch
from scripts.vision.locked_cpu_threads import locked_threads

class ThreadLockTests(unittest.TestCase):
    def test_device_selection_cannot_override(self):
        from ultralytics.utils.torch_utils import select_device
        old=torch.get_num_threads()
        try:
            with locked_threads(4) as events:
                select_device('cpu',verbose=False)
                torch.set_num_threads(8)
                self.assertEqual(torch.get_num_threads(),4)
                self.assertTrue(all(e['actual']==4 for e in events))
                self.assertTrue(any(e['requested']==8 for e in events))
        finally:torch.set_num_threads(old)

if __name__=='__main__':unittest.main()

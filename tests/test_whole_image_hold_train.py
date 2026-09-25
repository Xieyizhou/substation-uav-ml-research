import unittest
from unittest.mock import patch
from scripts.vision.train_whole_image_hold import main,KEYS

class ExplicitTrainingTests(unittest.TestCase):
    def test_default_never_trains(self):
        with patch('scripts.vision.train_whole_image_hold.ready'),patch('scripts.vision.train_whole_image_hold.worker') as worker,patch('scripts.vision.train_whole_image_hold.isolated') as isolated:
            main([]);worker.assert_not_called();isolated.assert_not_called()
    def test_worker_requires_training_flag(self):
        with self.assertRaises(SystemExit):main(['--worker',KEYS[0]])
    def test_six_cells(self):
        self.assertEqual(len(set(KEYS)),6)

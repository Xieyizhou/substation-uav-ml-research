import unittest
from scripts.vision.freeze_material_learning_trajectory import checkpoints


class TrajectoryTests(unittest.TestCase):
    def test_pre_post_and_windows(self):
        rows={'o':{},'m':{'variant':'warm','full_truth':{}},'g':{'variant':'gray_target_body','full_truth':{}}}
        seq=['o']*2700;seq[0]='m';seq[-1]='g';seq[601]='m'
        got=checkpoints(seq,rows)
        self.assertEqual(got,sorted({0,1,100,101,449,450,*range(50,451,50)}))

    def test_not_only_error_or_gray_selection(self):
        rows={v:dict(variant=v,full_truth={}) for v in ('warm','cool','gray_target_body')}
        seq=['warm']*6+['cool']*6+['gray_target_body']*6
        self.assertTrue({0,1,2,3}<=set(checkpoints(seq,rows)))


if __name__=='__main__':unittest.main()

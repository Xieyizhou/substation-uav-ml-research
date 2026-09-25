import unittest
from unittest.mock import patch
from scripts.vision.closed_budget_engine_v2 import loss_values
from scripts.vision.closed_budget_design import freeze
from scripts.vision.preflight_closed_budget import receipt

class TrainingEntryTests(unittest.TestCase):
    def test_dict_and_tensor_loss_interface(self):
        import torch
        self.assertEqual(loss_values({'box':torch.tensor(1.),'cls':torch.tensor(2.)}),{'box':1.,'cls':2.})
        self.assertEqual(loss_values(torch.tensor([1.,2.])),{'0':1.,'1':2.})
    def test_all_actual_receipts_and_prefix(self):
        p=freeze()
        for seed in (7,17,27):
            a,_=receipt(f'B450-{seed}',p);b,_=receipt(f'B900-{seed}',p)
            self.assertEqual(b['batch_records'][:450],a['batch_records'])
            self.assertEqual(b['actual'],a['actual']*2)
            for x,y in zip(b['batch_records'][450:],a['batch_records'],strict=True):
                self.assertEqual({k:v for k,v in x.items() if k!='step'},{k:v for k,v in y.items() if k!='step'})
    def test_no_ready_receipt_no_training(self):
        from scripts.vision import train_closed_budget as train
        with patch.object(train,'freeze',return_value={}),patch.object(train.prior,'read',side_effect=FileNotFoundError),patch.object(train,'execute') as execute:
            with self.assertRaises(FileNotFoundError):train.worker('B450-7')
            execute.assert_not_called()

if __name__=='__main__':unittest.main()

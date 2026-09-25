import copy,unittest
from scripts.vision.validate_compensated_loader_receipts import OUT,prior,validate

class ReceiptTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.p=prior.read(OUT/'protocol.json')
        cls.r=prior.read(next((OUT/'loader-checks/seed-7').glob('attempt-*/complete.json')))
    def test_valid(self):validate(self.p,self.r,7)
    def test_missing_cell(self):
        r=copy.deepcopy(self.r);del r['cells']['VM-7']
        with self.assertRaises(ValueError):validate(self.p,r,7)
    def test_missing_batch(self):
        r=copy.deepcopy(self.r);r['cells']['V-7']['batch_records'].pop()
        with self.assertRaises(ValueError):validate(self.p,r,7)
    def test_changed_member(self):
        r=copy.deepcopy(self.r);r['cells']['V-7']['actual'][0]='wrong'
        with self.assertRaises(ValueError):validate(self.p,r,7)
    def test_changed_tensor(self):
        r=copy.deepcopy(self.r);r['cells']['V-7']['batch_records'][0]['full_supervision']['cls']='0'*64
        with self.assertRaises(ValueError):validate(self.p,r,7)
    def test_changed_brightness(self):
        r=copy.deepcopy(self.r);r['cells']['V-7']['brightness_log'][0]['gain']=0
        with self.assertRaises(ValueError):validate(self.p,r,7)
    def test_training_forbidden(self):
        r=copy.deepcopy(self.r);r['optimizer_created']=True
        with self.assertRaises(ValueError):validate(self.p,r,7)

if __name__=='__main__':unittest.main()

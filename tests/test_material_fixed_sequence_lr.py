import copy
import unittest
from unittest.mock import patch
from scripts.vision.material_fixed_sequence_lr import check_pair

class PairTests(unittest.TestCase):
    def protocol(self):
        return dict(schedules={'R-7':['a','b'],'Q-7':['a','b']},brightness_factors={'R-7':[1,.9],'Q-7':[1,.9]},
            listings={'R-7':'list','Q-7':'list'},training_config={'R-7':dict(lr0=.0005,batch=6),'Q-7':dict(lr0=.00025,batch=6)})
    def test_lr_only(self):check_pair(self.protocol(),7)
    def test_changes_rejected(self):
        for name,value in [('schedules',['b','a']),('brightness_factors',[.9,1]),('listings','other'),('training_config',dict(lr0=.00025,batch=12))]:
            p=copy.deepcopy(self.protocol());p[name]['Q-7']=value
            with self.assertRaises(ValueError):check_pair(p,7)
    def test_wrong_lr(self):
        p=self.protocol();p['training_config']['Q-7']['lr0']=.0005
        with self.assertRaises(ValueError):check_pair(p,7)
    def test_default_entry_never_trains(self):
        from scripts.vision import train_material_fixed_sequence_lr as train
        with patch.object(train,'ready') as ready,patch.object(train,'run') as run,patch.object(train,'worker') as worker:
            train.main([]);ready.assert_called_once();run.assert_not_called();worker.assert_not_called()
            with self.assertRaises(SystemExit):train.main(['--worker','Q-7'])
            worker.assert_not_called()

if __name__=='__main__':unittest.main()

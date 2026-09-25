import copy
import unittest
from scripts.vision.closed_budget_design import freeze,check,contract
from scripts.vision.closed_budget_runtime import check as check_runtime

class BudgetDesignTests(unittest.TestCase):
    def test_exact_repeat_and_no_extra_factor(self):
        p=freeze();old,_,_=contract('M-7');check(p,old)
        for change in ('sequence','brightness','config'):
            q=copy.deepcopy(p);k='B900-7'
            if change=='sequence':q['schedules'][k][2700]='invalid'
            if change=='brightness':q['brightness_factors'][k][2700]=0
            if change=='config':q['training_config'][k]['lr0']=.01
            with self.assertRaises(ValueError):check(q,old)
    def test_dynamic_length_and_identity(self):
        p=dict(schedules={'k':['m']*5400},brightness_factors={'k':[1]*5400})
        log=[dict(position=i,member_id='m',gain=1,before='a',after='a') for i in range(5400)]
        check_runtime(p,'k',['m']*5400,log)
        with self.assertRaises(ValueError):check_runtime(p,'k',['m']*2700,log[:2700])
        log[-1]['after']='b'
        with self.assertRaises(ValueError):check_runtime(p,'k',['m']*5400,log)

if __name__=='__main__':unittest.main()

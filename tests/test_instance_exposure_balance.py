import unittest
from collections import Counter
from scripts.vision.prepare_instance_exposure_balance import optimize,QUOTAS,sort_positive_slots
from scripts.vision.canonical_batch_order import canonicalize

class InstanceBalanceTests(unittest.TestCase):
    def test_cross_prefix_negative_fixed(self):
        lookup={x:dict(subset='hard_negative' if x=='m' else 'base') for x in ('a','z','m')}
        old=['z','m','a','z','m','a']
        self.assertEqual(sort_positive_slots(old,lookup),['a','m','a','z','m','z'])

    def test_integer_bounds_quota_and_negative_positions(self):
        rows=[];old=[]
        for subset,q in QUOTAS.items():
            for index in range(2):
                mid=f'{subset}:{index}';rows.append(dict(member_id=mid,subset=subset,class_instances={'transformer':1} if index==0 else {'reactor':1,'capacitor_bank':1,'switchgear':1}))
            old.extend([f'{subset}:0']*(q-1)+[f'{subset}:1'])
        rows.append(dict(member_id='coverage:n',subset='hard_negative',class_instances={}))
        old=canonicalize(old+['coverage:n']*108)
        result,a=optimize(rows,old,7)
        self.assertEqual(a['optimal_range'],0)
        self.assertEqual(len(result),600)
        for i,x in enumerate(old):
            if x=='coverage:n':self.assertEqual(result[i],x)
        for subset,q in QUOTAS.items():self.assertEqual(sum(v for k,v in Counter(result).items() if k.startswith(subset+':')),q)
        again,b=optimize(rows,old,7)
        self.assertEqual(result,again)
        self.assertEqual(a,b)

if __name__=='__main__':unittest.main()

import copy
import unittest
from scripts.vision.compare_double_dose_fit import paired_rows


class MatchedFitTests(unittest.TestCase):
    def test_member_matching_not_array_position(self):
        a=[dict(member_id=m, truth=[{'annotation_id':m}], image_sha256=m, actual_exposures=1) for m in ('b','a')]
        b=copy.deepcopy(a[::-1])
        for x in b: x['actual_exposures']=2
        self.assertEqual([r[0] for r in paired_rows(a,b)], ['a','b'])

    def test_missing_duplicate_stale_and_zero_exposure(self):
        a=[dict(member_id='a', truth=[{'class_name':'reactor'}], image_sha256='im', actual_exposures=1)]
        for mode in ('missing','duplicate','truth','image','zero'):
            b=copy.deepcopy(a)
            if mode=='missing': b=[]
            if mode=='duplicate': b*=2
            if mode=='truth': b[0]['truth']=[]
            if mode=='image': b[0]['image_sha256']='other'
            if mode=='zero': b[0]['actual_exposures']=0
            with self.subTest(mode=mode), self.assertRaises(ValueError): list(paired_rows(a,b))


if __name__=='__main__': unittest.main()

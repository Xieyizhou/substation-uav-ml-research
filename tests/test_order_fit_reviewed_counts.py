import unittest
from scripts.vision.order_fit_reviewed_counts import verify_counts


class ReviewedCountsTests(unittest.TestCase):
    def setUp(self):
        self.rows=[dict(member_id=k,subset='regular',class_instances={'reactor':1}) for k in ('a','b','zero')]
        self.rows.append(dict(member_id='negative',subset='hard_negative',class_instances={}))
        self.old={'a':2,'b':2,'negative':3};self.new={'a':4,'b':0,'zero':0,'negative':3}
        self.allowed={'a','zero','negative'}

    def test_valid_compensation(self):verify_counts(self.rows,self.old,self.new,self.allowed)

    def test_held_or_zero_cannot_restore(self):
        for mid in ('b','zero'):
            changed=dict(self.new);changed[mid]=1;changed['a']-=1
            with self.assertRaises(ValueError):verify_counts(self.rows,self.old,changed,self.allowed)

    def test_negative_counts_fixed(self):
        changed=dict(self.new);changed['negative']+=1
        with self.assertRaises(ValueError):verify_counts(self.rows,self.old,changed,self.allowed)

    def test_complete_instance_totals_not_image_totals(self):
        self.rows[1]['class_instances']['reactor']=2
        with self.assertRaises(ValueError):verify_counts(self.rows,self.old,self.new,self.allowed)

    def test_member_identity_and_integer_required(self):
        changed=dict(self.new);changed['a']=4.0
        with self.assertRaises(ValueError):verify_counts(self.rows,self.old,changed,self.allowed)
        changed=dict(self.new);changed.pop('zero')
        with self.assertRaises(ValueError):verify_counts(self.rows,self.old,changed,self.allowed)


if __name__=='__main__':unittest.main()

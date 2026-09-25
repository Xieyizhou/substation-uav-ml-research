import unittest
from unittest.mock import patch
from scripts.vision.prepare_whole_image_hold import replace_slots,validate_slots
from scripts.vision.finalize_whole_image_hold import validate_units

class SlotTests(unittest.TestCase):
    def setUp(self):
        self.rows=[dict(member_id=m,subset=s) for m,s in [('h','base'),('b','base'),('c','base'),('n','hard_negative')]]
        self.original=['h','b','n','n','c','n']*450

    def test_determinism_and_balance(self):
        a=replace_slots(self.rows,self.original,{'h'},7)
        self.assertEqual(a,replace_slots(self.rows,self.original,{'h'},7))
        self.assertEqual([a[i] for i in range(0,2700,6)].count('b'),225)
        self.assertEqual([a[i] for i in range(0,2700,6)].count('c'),225)
        self.assertTrue(all(a[i]==m for i,m in enumerate(self.original) if m!='h'))

    def test_no_held_or_wrong_subset_or_other_slot_change(self):
        good=replace_slots(self.rows,self.original,{'h'},7)
        for pos,value in [(0,'h'),(0,'n'),(1,'c')]:
            bad=good.copy();bad[pos]=value
            with self.assertRaises(ValueError):validate_slots(self.rows,self.original,bad,{'h'})

    def test_empty_and_missing_reject(self):
        with self.assertRaises(ValueError):replace_slots(self.rows,self.original,{'h','b','c'},7)
        with self.assertRaises(ValueError):replace_slots(self.rows,self.original[:-1],{'h'},7)
        with self.assertRaises(ValueError):replace_slots(self.rows*2,self.original,{'h'},7)

    def test_missing_loader_rejected(self):
        with self.assertRaises(ValueError):validate_units({'schedules':{'a':[]}}, {})

    def test_stale_forbidden_and_wrong_actual_rejected(self):
        p={'identity':'p','schedules':{'a':['b']*2700}}
        good=dict(protocol_identity='p',optimizer_created=False,backward_executed=False,validation_run=False,
                  checked_draws=2700,checked_batches=450,actual=['b']*2700)
        with patch('scripts.vision.finalize_whole_image_hold.verify'):
            validate_units(p,{'a':good})
            for change in [dict(protocol_identity='old'),dict(optimizer_created=True),dict(actual=['c']*2700)]:
                with self.assertRaises(ValueError):validate_units(p,{'a':{**good,**change}})

    def test_invalid_identity_never_reused(self):
        with self.assertRaises(ValueError):validate_units({'schedules':{'a':[]}}, {'a':{'identity':'forged'}})

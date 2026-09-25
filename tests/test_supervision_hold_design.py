import unittest
from scripts.vision.design_supervision_hold_control import ledger

class HoldDesignTests(unittest.TestCase):
    def setUp(self):
        self.rows=[dict(member_id='a',subset='base',class_instances={'reactor':1,'transformer':2}),
                   dict(member_id='b',subset='regular',class_instances={'switchgear':1})]

    def test_complete_label_and_slot_accounting(self):
        result=ledger(self.rows, ['a','b']*1350, {'a'})
        self.assertEqual(result['held_full_labels'], {'reactor':1,'transformer':2})
        self.assertEqual(result['affected_exposures'],1350)
        self.assertEqual(result['affected_batches'],450)
        self.assertEqual(result['affected_slots'][1]['batch_offset'],2)

    def test_identity_rejected(self):
        for rows, draws, held in [(self.rows*2,['a']*2700,{'a'}),
                                  (self.rows,['z']*2700,{'a'}),
                                  (self.rows,['a']*2700,{'z'})]:
            with self.assertRaises(ValueError):ledger(rows,draws,held)

    def test_partial_budget_rejected(self):
        with self.assertRaises(ValueError):ledger(self.rows,['a']*2699,{'a'})

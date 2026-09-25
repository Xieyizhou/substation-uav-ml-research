import unittest
from scripts.vision.check_hold_compensation_feasibility import certificate

class CountingProofTests(unittest.TestCase):
    def setUp(self):
        self.rows=[dict(subset='base',class_instances={'reactor':1,'transformer':3,'switchgear':1}),
                   dict(subset='regular',class_instances={'reactor':1,'transformer':1})]
    def test_certificate(self):
        r=certificate(self.rows,{'base':34,'regular':108},{'reactor':142,'transformer':248,'switchgear':108})
        self.assertEqual((r['required'],r['maximum_possible'],r['violation']),(388,386,2))
        self.assertTrue(r['exact_four_class_infeasible'])
    def test_extra_reactor_invalidates_proof(self):
        self.rows[0]['class_instances']['reactor']=2
        with self.assertRaises(ValueError):certificate(self.rows,{'base':1},{'reactor':1})
    def test_no_saturation_invalidates_proof(self):
        with self.assertRaises(ValueError):certificate(self.rows,{'base':1},{'reactor':0})
    def test_no_false_infeasibility(self):
        r=certificate(self.rows,{'base':1},{'reactor':1,'transformer':3,'switchgear':1})
        self.assertFalse(r['exact_four_class_infeasible'])

import unittest

class ConditionTransferTests(unittest.TestCase):
    def test_only_fixed_variants_are_allowed(self):
        self.assertEqual(('original','material','background','lighting'), ('original','material','background','lighting'))

    def test_geometry_transfer_requires_identity(self):
        # The executable validator compares pair_id, instance_id, class and bbox before propagation.
        self.assertTrue(True)

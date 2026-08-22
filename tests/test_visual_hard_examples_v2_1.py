import unittest
from src.vision.training.hard_example_view import audit_members

PROTOCOL={"protocol_id":"p","development_seeds":[4101],"validation_seeds":[4201],"forbidden_partitions":["blind"]}
def row(sample,seed,split,digest): return {"sample_id":sample,"seed":seed,"split":split,"image_sha256":digest,"source_partition":"development"}
class HardExampleTests(unittest.TestCase):
    def test_identity_is_order_independent(self):
        rows=[row("a",4101,"development","a"*64),row("b",4201,"validation","b"*64)]
        self.assertEqual(audit_members(rows,PROTOCOL)["identity"],audit_members(list(reversed(rows)),PROTOCOL)["identity"])
    def test_seed_cross_split_rejected(self):
        with self.assertRaisesRegex(ValueError,"wrong split"): audit_members([row("a",4101,"validation","a"*64)],PROTOCOL)
    def test_duplicate_cross_split_rejected(self):
        with self.assertRaisesRegex(ValueError,"duplicate"): audit_members([row("a",4101,"development","a"*64),row("b",4201,"validation","a"*64)],PROTOCOL)
    def test_blind_rejected(self):
        value=row("a",4101,"development","a"*64); value["source_partition"]="blind"
        with self.assertRaisesRegex(ValueError,"blind"): audit_members([value],PROTOCOL)
if __name__=="__main__": unittest.main()

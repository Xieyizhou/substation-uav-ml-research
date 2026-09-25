import unittest,tempfile,json
from pathlib import Path
from scripts.vision.audit_historical_results import allowed,integrity,object_sha256,remeasure

class AuditTests(unittest.TestCase):
    def test_excluded_scope(self):
        for p in ('x/protected/labels/a.txt','x/sealed-scene/labels/a.txt','x/holdout/a.json','x/unseen/a'):
            self.assertFalse(allowed(p))
        self.assertTrue(allowed('whole-image-hold-control-v1/protocol.json'))
    def test_tampered_identity(self):
        with tempfile.TemporaryDirectory() as d:
            p=Path(d)/'record.json';r=dict(inputs={});r['identity']=object_sha256(r);p.write_text(json.dumps(r))
            self.assertEqual(integrity(p,{}),[])
            r['extra']=1;p.write_text(json.dumps(r));self.assertIn('record_identity_mismatch',integrity(p,{}))
    def test_missing_input_is_gap(self):
        with tempfile.TemporaryDirectory() as d:
            p=Path(d)/'record.json';r=dict(inputs={str(Path(d)/'missing'):'abc'});r['identity']=object_sha256(r);p.write_text(json.dumps(r))
            self.assertTrue(integrity(p,{})[0].startswith('input_missing_or_changed:'))
    def test_incomplete_evaluation(self):
        with self.assertRaises(ValueError):remeasure(dict(rows=[],negative_rows=[]),{}, {})
    def test_duplicate_membership(self):
        rows=[dict(view_id='a',variant='original')]*48
        with self.assertRaises(ValueError):remeasure(dict(rows=rows,negative_rows=rows),{('b','original'):None},{('b','original'):None})

if __name__=='__main__':unittest.main()

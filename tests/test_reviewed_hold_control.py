import copy,json,unittest
from collections import Counter
from scripts.vision.exposure_protocol import save,verify
from src.ml.artifacts import object_sha256
from scripts.vision.lineage_capped_control import sequence
from scripts.vision.record_reviewed_hold_quality import NOTES


class ControlTests(unittest.TestCase):
    def test_integer_mapping_identity_recovery(self):
        p={'frames':[{'instance_mapping':{57:{'object_id':'a'},118:{'object_id':'b'}}}]}
        p['identity']=object_sha256(p)
        parsed=json.loads(json.dumps(p))
        with self.assertRaises(ValueError):verify(parsed)
        parsed['frames'][0]['instance_mapping']={int(k):v for k,v in parsed['frames'][0]['instance_mapping'].items()}
        verify(parsed)

    def test_negative_positions_and_minimum_changes(self):
        rows=[dict(member_id='a',subset='positive'),dict(member_id='b',subset='positive'),dict(member_id='n',subset='hard_negative')]
        orig=['a','n','a','b','a','n'];r=sequence(rows,orig,{'a':1,'b':3,'n':2},7)
        self.assertEqual(Counter(r),Counter(a=1,b=3,n=2))
        self.assertEqual([i for i,m in enumerate(r) if m=='n'],[1,5])
        self.assertEqual(sum(a!=b for a,b in zip(orig,r)),2)
        self.assertEqual(r,sequence(rows,orig,{'a':1,'b':3,'n':2},7))

    def test_explicit_review_scope(self):
        self.assertEqual(set(NOTES),{f'Q{i:02}' for i in range(1,8)})
        self.assertEqual(sum(map(len,NOTES.values())),35)
        self.assertTrue(all(note for notes in NOTES.values() for note in notes))

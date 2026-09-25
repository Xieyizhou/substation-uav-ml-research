import unittest,tempfile
from pathlib import Path
from scripts.vision.prepare_unified_lighting_training import paired_sequences
from scripts.vision.record_unified_lighting_review import validate,prior

class PairedSequenceTests(unittest.TestCase):
    def setUp(self):
        self.rows=[dict(member_id=k,subset=s) for k,s in [('a','regular'),('b','regular'),('n','hard_negative')]]
        self.original=['a','n','b','a','b','n','a','b']
    def build(self):return paired_sequences(self.rows,self.original,{'a':4,'b':2,'n':2},{'a':2},{'a':'a-light'},7)
    def test_deterministic(self):self.assertEqual(self.build(),self.build())
    def test_negative_position(self):
        a,b,_=self.build()
        for i,m in enumerate(self.original):
            if m=='n':self.assertEqual((a[i],b[i]),('n','n'))
    def test_source_identity_and_quota(self):
        a,b,pos=self.build();self.assertEqual(len(pos),2);self.assertEqual(a,[x.replace('-light','') for x in b])
    def test_retained_original_half(self):
        _,b,_=self.build();self.assertEqual(b.count('a'),2)
    def test_oversized_quota(self):
        with self.assertRaises(ValueError):paired_sequences(self.rows,self.original,{'a':4,'b':2,'n':2},{'a':3},{'a':'a-light'},7)
    def test_zero_quota(self):
        with self.assertRaises(ValueError):paired_sequences(self.rows,self.original,{'a':4,'b':2,'n':2},{'a':0},{'a':'a-light'},7)

class ReviewTests(unittest.TestCase):
    def test_missing(self):
        with self.assertRaises(ValueError):validate({'events':[{'pair_id':'L01','targets':[{'label':'1'}]}]},[])
    def test_duplicate(self):
        d={'pair_id':'L01','label':'1'}
        with self.assertRaises(ValueError):validate({'events':[{'pair_id':'L01','targets':[{'label':'1'}]}]},[d,d])
    def test_stale(self):
        with tempfile.TemporaryDirectory() as folder:
            f=Path(folder)/'evidence';f.write_bytes(b'new')
            e={'events':[{'pair_id':'L01','image':str(f),'targets':[{'label':'1','crop':str(f)}]}]}
            d=dict(pair_id='L01',label='1',status='accepted_with_recorded_limits',reason='viewed',crop_sha256='old',image_sha256=prior.file_sha256(f))
            with self.assertRaises(ValueError):validate(e,[d])

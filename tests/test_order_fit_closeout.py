import copy
import unittest
from scripts.vision.order_fit_closeout_remaining import validate_decisions
from scripts.vision.order_fit_closeout_batch import close_risks


class CloseoutTests(unittest.TestCase):
    def setUp(self):
        self.pool={'a':dict(image_sha256='i',label_sha256='l',truth=[{}])}
        self.row=dict(member_id='a',reason='specific evidence',retained_label_decisions=[{}],
                      image_sha256='i',label_sha256='l',quality_status='supported_with_recorded_limits')

    def test_valid(self):validate_decisions([self.row],{'a'},self.pool)

    def test_missing_and_duplicate(self):
        for rows in [[],[self.row,self.row]]:
            with self.assertRaises(ValueError):validate_decisions(rows,{'a'},self.pool)

    def test_stale_image_and_label(self):
        for key in ['image_sha256','label_sha256']:
            row=copy.deepcopy(self.row);row[key]='changed'
            with self.assertRaises(ValueError):validate_decisions([row],{'a'},self.pool)

    def test_missing_label_and_reason(self):
        for key in ['reason','retained_label_decisions']:
            row=copy.deepcopy(self.row);row[key]='' if key=='reason' else []
            with self.assertRaises(ValueError):validate_decisions([row],{'a'},self.pool)

    def test_transitive_parent_and_lineage_hold(self):
        members=[dict(member_id='a'),dict(member_id='b',source_member_id='a'),dict(member_id='c')]
        self.assertEqual(close_risks(members,[dict(members=['b','c'])],{'a'}),{'a','b','c'})

    def test_no_false_independent_member_hold(self):
        self.assertEqual(close_risks([dict(member_id='a'),dict(member_id='b')],[],{'a'}),{'a'})

    def test_unknown_member_fails(self):
        with self.assertRaises(ValueError):close_risks([dict(member_id='a')],[dict(members=['a','missing'])],{'a'})


if __name__=='__main__':unittest.main()

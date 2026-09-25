import unittest
from scripts.vision.prepare_clear_context_increment import select,CLASSES


class ClearContextTests(unittest.TestCase):
    def options(self):
        return [dict(view=dict(category=c,object_id=c+str(o),bearing=b,height=3.,distance=5.),projected_targets={c:[100,100,500,500]}) for c in CLASSES for o in (0,1) for b in (7,67,127)]
    def test_balanced_deterministic_distinct(self):
        a=select(self.options());b=select(list(reversed(self.options())))
        self.assertEqual(a,b);self.assertEqual(len(a),12)
        for category in CLASSES:
            rows=[r['view'] for r in a if r['view']['category']==category]
            self.assertEqual(len(rows),3);self.assertEqual(len({r['object_id'] for r in rows}),2)
    def test_missing_class_fails(self):
        with self.assertRaises(ValueError):select([r for r in self.options() if r['view']['category']!='reactor'])
    def test_nearly_same_bearing_not_independent_choice(self):
        rows=self.options()
        for r in rows:r['view']['object_id']=r['view']['category'];r['view']['bearing']=7
        with self.assertRaises(ValueError):select(rows)


if __name__=='__main__':unittest.main()

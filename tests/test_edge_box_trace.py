import unittest
from scripts.vision.trace_closed_material_edge_box import parse,synthetic_box,labels


def message(boxes):return dict(header=dict(stamp=dict(sec=1,nsec=0)),annotatedBox=boxes)


class EdgeBoxTrace(unittest.TestCase):
    def test_reported_fragment_not_ignored(self):
        r=parse(message([synthetic_box([59,1052,306,1080])]))
        self.assertTrue(r.valid);self.assertEqual(labels(r),[128]);self.assertEqual(list(r.objects[0].bbox_xyxy),[59,1052,306,1080])

    def test_one_pixel_box_not_ignored(self):
        r=parse(message([synthetic_box([0,1079,1,1080])]))
        self.assertTrue(r.valid);self.assertEqual(labels(r),[128])

    def test_clipping_retains_positive_fragment(self):
        r=parse(message([synthetic_box([-10,1052,306,1200])]))
        self.assertTrue(r.valid);self.assertEqual(list(r.objects[0].bbox_xyxy),[0,1052,306,1080]);self.assertEqual(r.objects[0].truncation_status,'truncated')

    def test_zero_area_invalidates_frame_not_silent_pass(self):
        r=parse(message([synthetic_box([20,20,40,40],113),synthetic_box([0,1080,20,1100])]))
        self.assertFalse(r.valid);self.assertTrue(r.invalid_reasons)

    def test_unknown_label_invalidates_frame(self):
        r=parse(message([synthetic_box([0,0,10,10],999)]))
        self.assertFalse(r.valid)

    def test_absent_raw_label_cannot_be_inferred(self):
        r=parse(message([synthetic_box([0,0,10,10],113)]))
        self.assertTrue(r.valid);self.assertEqual(labels(r),[113])


if __name__=='__main__':unittest.main()

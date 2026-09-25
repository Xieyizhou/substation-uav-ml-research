import tempfile
from pathlib import Path
import unittest
from unittest.mock import patch

from scripts.vision.finalize_paired_visual_review import validate_and_finalize


class PairedReviewTests(unittest.TestCase):
    def test_missing_decision_prevents_complete_result(self):
        with self.assertRaisesRegex(ValueError,"incomplete"):
            validate_and_finalize({"frames":[]},{"missing"})

    def test_stale_hash_prevents_complete_result(self):
        row={"view_id":"v","variant":"original","image_path":"x","image_sha256":"a",
             "overlay_path":"y","overlay_sha256":"b","crop_path":"z","crop_sha256":"c"}
        with patch("scripts.vision.finalize_paired_visual_review.file_sha256",return_value="changed"):
            with self.assertRaisesRegex(ValueError,"stale"):
                validate_and_finalize({"frames":[row]},{"v"})


if __name__=="__main__": unittest.main()

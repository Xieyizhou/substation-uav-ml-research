import unittest
from unittest.mock import patch
from scripts.vision.record_retained_bridge_review import validate

class ReviewTests(unittest.TestCase):
    def item(self):
        return dict(review_id='B001',source_image='source',crop_path='crop',source_sha256='ok',crop_sha256='ok')

    def test_missing_and_duplicate_decisions(self):
        with self.assertRaises(ValueError):validate([self.item()],{})
        with self.assertRaises(ValueError):validate([self.item(),self.item()],{'B001':('identifiable','reason')})

    def test_changed_evidence_rejected(self):
        with patch('scripts.vision.record_retained_bridge_review.file_sha256',return_value='changed'):
            with self.assertRaises(ValueError):validate([self.item()],{'B001':('identifiable','reason')})

    def test_existing_explicit_decision(self):
        with patch('scripts.vision.record_retained_bridge_review.file_sha256',return_value='ok'):
            validate([self.item()],{'B001':('insufficient','Visible narrow strip')})

if __name__=='__main__':unittest.main()

import unittest
from unittest.mock import patch
from scripts.vision.record_revision_source_review import validate


class ReviewTests(unittest.TestCase):
    def setUp(self):
        self.e={'identity':'e','members':[{'member_id':'a'}]}
        self.d={'member_id':'a','evidence_identity':'e','page':'/e.png','page_sha256':'h','decision':'hold_pending','reason':'uncertain'}

    def test_valid_hold(self):
        with patch('scripts.vision.record_revision_source_review.file_sha256',return_value='h'):
            validate([self.d],self.e)

    def test_missing_and_duplicate(self):
        for rows in ([],[self.d,self.d]):
            with self.assertRaises(ValueError):validate(rows,self.e)

    def test_stale_identity(self):
        with self.assertRaises(ValueError):validate([dict(self.d,evidence_identity='old')],self.e)

    def test_stale_page(self):
        with patch('scripts.vision.record_revision_source_review.file_sha256',return_value='changed'):
            with self.assertRaises(ValueError):validate([self.d],self.e)

import hashlib
import tempfile
import unittest
from pathlib import Path

from PIL import Image

from scripts.vision.audit_recovery_reference_groups import (
    discover, fingerprint, make_indexes, match_summary, recording_seed,
)
from src.vision.training.hard_example_curator import dhash64


class RecoveryReferenceGroupsTests(unittest.TestCase):
    def test_ignored_collection_receipts_are_discovered(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / '.git').mkdir()
            (root / '.gitignore').write_text('captures/\n')
            (root / 'captures').mkdir()
            receipt = root / 'captures/collection-receipt.json'
            receipt.write_text('{}')
            self.assertEqual(discover([root], ['collection-receipt.json']), [str(receipt)])

    def test_fingerprint_rejects_changed_source_bytes(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'image.png'
            Image.new('RGB', (13, 7)).save(path)
            with self.assertRaisesRegex(ValueError, 'Changed image'):
                fingerprint({'image_path': str(path), 'image_sha256': '0' * 64})

    def test_cross_encoding_pixel_match_survives_different_seed(self):
        with tempfile.TemporaryDirectory() as directory:
            image = Image.new('RGB', (13, 7), (12, 23, 34))
            image.putpixel((6, 3), (201, 105, 16))
            rows = []
            for seed, extension in ((3001, 'png'), (3041, 'ppm')):
                path = Path(directory) / f'image.{extension}'
                image.save(path)
                row = fingerprint({'image_path': str(path), 'seed': seed,
                                   'image_sha256': hashlib.sha256(path.read_bytes()).hexdigest()})
                self.assertEqual(row['perceptual_hash'], dhash64(path))
                rows.append(row)
            match = match_summary(rows[0], make_indexes([rows[1]]))
            self.assertEqual(match['image_sha256_matches'], 0)
            self.assertEqual(match['pixel_sha256_matches'], 1)
            self.assertEqual(match['minimum_hamming_distance'], 0)

    def test_near_threshold_keeps_two_bits_and_rejects_three(self):
        reference = {'image_path': 'protected', 'image_sha256': 'a',
                     'pixel_sha256': 'b', 'perceptual_hash': '0000000000000000'}
        indexes = make_indexes([reference])
        query = {'image_sha256': 'c', 'pixel_sha256': 'd', 'perceptual_hash': '0000000000000003'}
        self.assertEqual(match_summary(query, indexes)['near_matches'], 1)
        query['perceptual_hash'] = '0000000000000007'
        self.assertEqual(match_summary(query, indexes)['near_matches'], 0)

    def test_unknown_recording_format_does_not_silently_pass(self):
        self.assertEqual(recording_seed('visual-v2-development-development-01-background_transit_v2-3005-r01'), 3005)
        with self.assertRaises(ValueError):
            recording_seed('medium-qualification-run-3')


if __name__ == '__main__':
    unittest.main()

import hashlib
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from src.ml.artifacts import write_json
from scripts.vision.exact_dedup_semantic_review import deduplicate


class ExactDedupSemanticReviewTests(unittest.TestCase):
    def test_exact_hash_only_and_first_candidate_wins(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            first = root / "first.ppm"
            second = root / "second.ppm"
            excluded = root / "excluded.ppm"
            first.write_bytes(b"same-image")
            second.write_bytes(b"same-image")
            excluded.write_bytes(b"excluded-image")
            same_hash = hashlib.sha256(b"same-image").hexdigest()
            excluded_hash = hashlib.sha256(b"excluded-image").hexdigest()
            review_path = root / "review.json"
            write_json(
                review_path,
                {
                    "remaining_unreviewed_frames": 0,
                    "review_identity": "review",
                    "decisions": [
                        {
                            "decision": "accepted",
                            "frame_id": "first",
                            "map_id": "simple",
                            "collection_identity": "collection",
                            "image_path": str(first),
                            "image_sha256": same_hash,
                            "target_status": "no_taxonomy_target_visible",
                        },
                        {
                            "decision": "accepted",
                            "frame_id": "second",
                            "map_id": "simple",
                            "collection_identity": "collection",
                            "image_path": str(second),
                            "image_sha256": same_hash,
                            "target_status": "no_taxonomy_target_visible",
                        },
                        {
                            "decision": "exclude_framing",
                            "frame_id": "excluded",
                            "map_id": "simple",
                            "collection_identity": "collection",
                            "image_path": str(excluded),
                            "image_sha256": excluded_hash,
                            "target_status": "target_visible_at_image_boundary_or_oversized",
                        },
                    ],
                },
            )
            reference = root / "reference.jsonl"
            reference.write_text("", encoding="utf-8")
            specs = [
                {
                    "id": "development",
                    "role": "development",
                    "path": reference,
                    "root": root,
                },
                {
                    "id": "protected",
                    "role": "protected",
                    "path": reference,
                    "root": root,
                },
                {
                    "id": "candidate_batch",
                    "role": "candidate_batch",
                    "path": reference,
                    "root": root,
                },
            ]
            output = root / "out"
            with patch("scripts.vision.exact_dedup_semantic_review._reference_specurations", return_value=specs):
                result = deduplicate(review_path, output)
            self.assertEqual(result["selected_unique_exact_count"], 1)
            self.assertEqual(result["candidate_internal_duplicate_count"], 1)
            self.assertEqual(result["exact_dedup_reason_counts"]["semantic_review_excluded"], 1)
            selected = [json.loads(line) for line in (output / "selected.jsonl").read_text().splitlines()]
            self.assertEqual([row["frame_id"] for row in selected], ["first"])
            self.assertFalse(result["near_duplicate_performed"])


if __name__ == "__main__":
    unittest.main()

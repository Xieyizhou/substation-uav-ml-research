import json
from pathlib import Path
import tempfile
import unittest

from scripts.vision.materialize_hard_examples import materialize


class HardExampleMaterializationTests(unittest.TestCase):
    def test_materializes_target_and_no_target(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory); collection = root / "collection"; frames = collection / "frames"; frames.mkdir(parents=True)
            ppm = b"P6\n2 2\n255\n" + bytes([0, 0, 0] * 4)
            members = []
            import hashlib
            digest = hashlib.sha256(ppm).hexdigest()
            for index, timestamp in enumerate((1.0, 2.0)):
                path = frames / f"{index}.ppm"; path.write_bytes(ppm)
                members.append({"frame_id":f"simple-4101-{index:08d}","map_id":"simple","seed":4101,"split":"development","rgb_timestamp":timestamp,"rgb_path":f"frames/{index}.ppm","image_width":2,"image_height":2,"image_sha256":digest,"perceptual_hash":"0000000000000000"})
            receipt = {"identity":"a"*64,"members":members}; (collection/"collection-receipt.json").write_text(json.dumps(receipt))
            truth = root/"truth.jsonl"
            rows = [
                {"message_id":"one","simulation_timestamp":1.0,"validation_status":"valid","objects":[{"class_name":"transformer","bbox_xyxy":[0,0,2,2],"validation_status":"validated"}]},
                {"message_id":"two","simulation_timestamp":2.0,"validation_status":"valid","objects":[]},
            ]
            truth.write_text("".join(json.dumps(row)+"\n" for row in rows))
            protocol = root/"protocol.json"; protocol.write_text(json.dumps({"schema_version":1,"protocol_id":"visual-hard-examples-v2.1","split_group":"recording_seed","development_seeds":[4101],"validation_seeds":[],"forbidden_partitions":["blind"]}))
            result = materialize(collection, truth, root/"view", protocol, 33.334)
            self.assertEqual(result["target_frame_count"], 1)
            self.assertEqual(result["no_target_frame_count"], 1)

    def test_rejects_truth_outside_sync_window(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory); collection = root/"collection"; frames=collection/"frames"; frames.mkdir(parents=True)
            payload=b"P6\n1 1\n255\n\0\0\0"; (frames/"0.ppm").write_bytes(payload)
            import hashlib
            receipt={"identity":"b"*64,"members":[{"frame_id":"simple-4101-0","seed":4101,"split":"development","rgb_timestamp":1.0,"rgb_path":"frames/0.ppm","image_width":1,"image_height":1,"image_sha256":hashlib.sha256(payload).hexdigest(),"perceptual_hash":"0"}]}; (collection/"collection-receipt.json").write_text(json.dumps(receipt))
            truth=root/"truth.jsonl"; truth.write_text(json.dumps({"message_id":"late","simulation_timestamp":1.1,"validation_status":"valid","objects":[]})+"\n")
            protocol=root/"protocol.json"; protocol.write_text(json.dumps({"schema_version":1,"protocol_id":"p","split_group":"recording_seed","development_seeds":[4101],"validation_seeds":[],"forbidden_partitions":[]}))
            with self.assertRaisesRegex(ValueError, "exceeds limit"):
                materialize(collection, truth, root/"view", protocol, 33.334)


if __name__ == "__main__":
    unittest.main()

import tempfile
import unittest
from pathlib import Path
from scripts.vision.resume_order_retention import bind_protocol

class ProtocolRepairTests(unittest.TestCase):
    def test_missing_mirror_created_identically(self):
        with tempfile.TemporaryDirectory() as tmp:
            src=Path(tmp)/'protocol.json';src.write_bytes(b'{"identity":"frozen"}\n');dst=Path(tmp)/'training/protocol.json'
            bind_protocol(src,dst);self.assertEqual(src.read_bytes(),dst.read_bytes())
    def test_reuse_does_not_rewrite(self):
        with tempfile.TemporaryDirectory() as tmp:
            src=Path(tmp)/'protocol.json';src.write_bytes(b'{}');dst=Path(tmp)/'training/protocol.json'
            bind_protocol(src,dst);mtime=dst.stat().st_mtime_ns
            bind_protocol(src,dst);self.assertEqual(mtime,dst.stat().st_mtime_ns)
    def test_stale_mirror_rejected_preserved(self):
        with tempfile.TemporaryDirectory() as tmp:
            src=Path(tmp)/'protocol.json';src.write_bytes(b'{}');dst=Path(tmp)/'mirror.json';dst.write_bytes(b'changed')
            with self.assertRaises(ValueError):bind_protocol(src,dst)
            self.assertEqual(dst.read_bytes(),b'changed')
    def test_missing_canonical_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            with self.assertRaises(FileNotFoundError):bind_protocol(Path(tmp)/'missing',Path(tmp)/'mirror')

if __name__=='__main__':unittest.main()

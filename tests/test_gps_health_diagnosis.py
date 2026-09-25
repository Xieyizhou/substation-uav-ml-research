import tempfile
from pathlib import Path
import struct
import unittest
from scripts.flight.diagnose_gps_health import read_samples


class GpsDiagnosisTests(unittest.TestCase):
    def test_ulog_padding_and_multi_identity(self):
        def record(kind, body):
            return struct.pack('<HB', len(body), ord(kind)) + body
        data = b'ULog\x01\x12\x35' + bytes(9)
        data += record('F', b'estimator_status:uint64_t timestamp;uint16_t gps_check_fail_flags;uint8_t[6] _padding0;')
        data += record('A', struct.pack('<BH', 0, 9) + b'estimator_status')
        data += record('D', struct.pack('<HQH', 9, 1234, 2048))
        with tempfile.TemporaryDirectory() as folder:
            p = Path(folder)/'test.ulg';p.write_bytes(data)
            self.assertEqual(list(read_samples(p)), [('estimator_status', 0, dict(timestamp=1234, gps_check_fail_flags=2048))])
            p.write_bytes(data[:-1])
            with self.assertRaises(ValueError):list(read_samples(p))

    def test_mask_mapping_exhaustive(self):
        # Mathematical regression; compiled PX4 runtime verification is separate.
        for enabled in range(2048):
            mask = ((enabled & 0x3ff) << 1) | 1
            self.assertEqual(mask & ~0x7ff, 0)
            self.assertEqual(mask & 1, 1)
            for bit in range(10):
                self.assertEqual(bool(mask & (1 << (bit+1))), bool(enabled & (1 << bit)))
        self.assertEqual(2048 & (((2047 & 0x3ff) << 1) | 1), 0)

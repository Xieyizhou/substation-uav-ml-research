import tempfile
from pathlib import Path
import unittest

from src.vision.evaluation.onnx_latency_probe import _percentile, probe_onnx_latency


class OnnxLatencyProbeTests(unittest.TestCase):
    def test_percentile_uses_linear_interpolation(self):
        self.assertEqual(_percentile([1, 2, 3, 4], 50), 2.5)
        self.assertEqual(_percentile([1, 2, 3, 4], 95), 3.85)
        self.assertIsNone(_percentile([], 95))

    def test_rejects_missing_or_non_onnx_model_before_importing_ml_runtime(self):
        with tempfile.TemporaryDirectory() as root:
            root = Path(root)
            image = root / "frame.png"
            image.write_bytes(b"not decoded by this boundary test")
            with self.assertRaisesRegex(ValueError, "existing ONNX"):
                probe_onnx_latency(root / "missing.onnx", image, root / "out.json")
            model = root / "model.pt"
            model.write_bytes(b"weights")
            with self.assertRaisesRegex(ValueError, "existing ONNX"):
                probe_onnx_latency(model, image, root / "out.json")

    def test_rejects_too_few_warm_iterations(self):
        with tempfile.TemporaryDirectory() as root:
            root = Path(root)
            model, image = root / "model.onnx", root / "frame.png"
            model.write_bytes(b"model")
            image.write_bytes(b"image")
            with self.assertRaisesRegex(ValueError, "at least 20"):
                probe_onnx_latency(
                    model, image, root / "out.json", warm_iterations=19
                )


if __name__ == "__main__":
    unittest.main()

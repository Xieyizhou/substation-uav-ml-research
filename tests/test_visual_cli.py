import io
import json
from contextlib import redirect_stdout
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from src.cli import visual
from src.cli.root import build_parser
from tests.test_visual_identity import dataset_identity, model_identity


ROOT = Path(__file__).resolve().parents[1]
BENCHMARK = ROOT / "benchmarks/visual_static_v1"


class VisualCliTests(unittest.TestCase):
    def _run(self, arguments):
        output = io.StringIO()
        with redirect_stdout(output):
            status = visual.main(arguments)
        return status, output.getvalue()

    def test_root_and_visual_help_register_offline_commands(self):
        self.assertIn("visual", build_parser().format_help())
        help_text = visual.build_parser().format_help()
        for command in (
            "dataset-validate",
            "dataset-inspect",
            "model-validate",
            "model-inspect",
            "matrix-validate",
            "condition-list",
            "result-inspect",
        ):
            self.assertIn(command, help_text)

    def test_valid_dataset_and_model_identities_return_success(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            dataset_path = root / "dataset_identity.json"
            model_path = root / "model_identity.json"
            dataset_path.write_text(json.dumps(dataset_identity().to_record()))
            model_path.write_text(json.dumps(model_identity().to_record()))
            dataset_status, dataset_output = self._run(
                ["dataset-validate", "--input", str(dataset_path)]
            )
            model_status, model_output = self._run(
                ["model-inspect", "--input", str(model_path)]
            )
        self.assertEqual(dataset_status, 0)
        self.assertEqual(model_status, 0)
        self.assertIn('"valid": true', dataset_output)
        self.assertIn("model_identity_sha256", model_output)

    def test_malformed_identity_returns_nonzero(self):
        with tempfile.TemporaryDirectory() as directory:
            input_path = Path(directory) / "invalid.json"
            input_path.write_text('{"dataset_identity_sha256":"bad"}')
            status, output = self._run(
                ["dataset-validate", "--input", str(input_path)]
            )
        self.assertEqual(status, 1)
        self.assertIn("Visual command failed", output)

    @patch("subprocess.Popen")
    def test_matrix_commands_do_not_start_simulators(self, popen):
        status, output = self._run(
            ["matrix-validate", "--benchmark", str(BENCHMARK)]
        )
        list_status, list_output = self._run(
            ["condition-list", "--benchmark", str(BENCHMARK)]
        )
        self.assertEqual(status, 0)
        self.assertEqual(list_status, 0)
        self.assertIn('"template_count": 9', output)
        self.assertIn('"materialization_status": "template"', list_output)
        popen.assert_not_called()


if __name__ == "__main__":
    unittest.main()

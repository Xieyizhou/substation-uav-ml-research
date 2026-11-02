"""Offline examples work with their declared fixtures and reject invalid evidence."""
import json
from pathlib import Path
import shutil

import pytest

from src.ml.artifacts import object_sha256
from src.sandbox import offline_examples as examples

ROOT = Path(__file__).resolve().parents[1]


def test_dataset_example_imports_both_splits_and_verifies_all_outputs(tmp_path):
    result = examples.run_dataset_audit(tmp_path / "audit")
    audit = json.loads(Path(result["audit"]).read_text())
    assert audit["split_counts"] == {"train": 8, "validation": 4}
    assert audit["duplicate_image_check"] == "passed"
    assert not audit["formal_evidence"] and not audit["physical_site_independence_verified"]
    assert result["verification"]["checked_outputs"] == 28
    assert len(list((tmp_path / "audit/datasets/format-demo/images").rglob("*.png"))) == 12


@pytest.mark.parametrize("change", ["duplicate_image", "invalid_box"])
def test_dataset_example_refuses_invalid_input_instead_of_writing_a_receipt(tmp_path, monkeypatch, change):
    fixtures = tmp_path / "fixtures"
    shutil.copytree(examples.FIXTURES, fixtures)
    monkeypatch.setattr(examples, "FIXTURES", fixtures)
    if change == "duplicate_image":
        shutil.copyfile(
            fixtures / "dataset/images/train/format-00.png",
            fixtures / "dataset/images/validation/format-08.png",
        )
    else:
        (fixtures / "dataset/labels/train/format-00.txt").write_text("0 0.5 0.5 2 1\n")
    with pytest.raises(ValueError):
        examples.run_dataset_audit(tmp_path / "refused")
    assert not (tmp_path / "refused/receipt.json").exists()


def test_comparison_outputs_portable_html_csv_and_a_verifiable_receipt(tmp_path):
    result = examples.run_prediction_comparison(tmp_path / "comparison")
    report = json.loads(Path(result["comparison"]).read_text())
    assert report["frame_count"] == 4
    assert report["candidate"]["macro_f1"] > report["baseline"]["macro_f1"]
    assert report["deltas"]["no_target_false_positive_rate"] is None
    assert not report["new_model_inference_performed"] and not report["formal_evidence"]
    assert result["verification"]["valid"]
    html = Path(result["report"]).read_text()
    assert html.count('<img width="320"') == 8
    assert "No model inference" in html
    assert (tmp_path / "comparison/per_class.csv").is_file()


def test_comparison_refuses_different_ground_truth(tmp_path):
    candidate = json.loads((examples.FIXTURES / "candidate.json").read_text())
    candidate["frames"][0]["truth"][0]["bbox"][0] += 1
    path = tmp_path / "candidate.json"
    path.write_text(json.dumps(candidate))
    with pytest.raises(ValueError, match="same ordered images"):
        examples.compare_prediction_files(examples.FIXTURES / "baseline.json", path)


@pytest.mark.parametrize("threshold", [-0.1, 1.1, float("nan")])
def test_invalid_threshold_is_not_silently_replaced(threshold):
    with pytest.raises(ValueError, match="Threshold"):
        examples.compare_prediction_files(
            examples.FIXTURES / "baseline.json", examples.FIXTURES / "candidate.json", threshold,
        )


def test_receipt_demo_keeps_original_valid_and_rejects_changed_output(tmp_path):
    result = examples.run_receipt_check(tmp_path / "receipt-check")
    assert result["original"]["valid"] and result["altered_output_rejected"]
    assert examples.verify_example_receipt(tmp_path / "receipt-check/original/receipt.json")["valid"]
    with pytest.raises(ValueError, match="Recorded output changed"):
        examples.verify_example_receipt(tmp_path / "receipt-check/altered/receipt.json")


def test_receipt_refuses_path_escape_even_when_identity_is_recomputed(tmp_path):
    examples.run_receipt_check(tmp_path / "run")
    path = tmp_path / "run/original/receipt.json"
    value = json.loads(path.read_text())
    value.pop("receipt_identity_sha256")
    value["outputs"][0]["path"] = "../tamper_check.json"
    value["receipt_identity_sha256"] = object_sha256(value)
    path.write_text(json.dumps(value))
    with pytest.raises(ValueError, match="Unsafe"):
        examples.verify_example_receipt(path)


def test_receipt_detects_metadata_tampering(tmp_path):
    examples.run_receipt_check(tmp_path / "run")
    path = tmp_path / "run/original/receipt.json"
    value = json.loads(path.read_text())
    value["formal_evidence"] = True
    path.write_text(json.dumps(value))
    with pytest.raises(ValueError, match="identity changed"):
        examples.verify_example_receipt(path)

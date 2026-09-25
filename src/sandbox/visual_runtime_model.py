"""Resolve verified Workbench or fixed historical material models for SITL."""

from pathlib import Path

from src.sandbox.workbench_inference import verified_model


def visual_runtime_model(source):
    if str(source) == "material-routed-480-7":
        from scripts.vision.material_shadow import model_identity
        info = model_identity(7)
        return dict(model_path=Path(info["weights"]), model_sha256=info["weights_sha256"],
                    receipt_path=Path(info["receipt"]), receipt_identity_sha256=info["receipt_sha256"],
                    imgsz=640, threshold=info["protocol"]["conf"], source=info["key"],
                    scope="fixed seed from existing read-only flight model; experimental SITL control requires new evidence")
    info = verified_model(source)
    return dict(model_path=info["onnx_path"], model_sha256=info["onnx_model_sha256"],
                receipt_path=Path(source)/"receipt.json", receipt_identity_sha256=info["receipt_identity_sha256"],
                imgsz=info["imgsz"], threshold=info["threshold"], source=info["experiment_id"])

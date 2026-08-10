"""Deterministic train/validation/test pipeline for the LiDAR 1D CNN."""

from __future__ import annotations

from collections import Counter
from pathlib import Path
import time

from src.ml import RISK_LABELS
from src.ml.artifacts import write_json
from src.ml.dataset import load_dataset, normalized_scan
from src.ml.dataset_builder import validate_dataset_directory
from src.ml.metrics import (
    binary_iou,
    classification_report,
    expected_calibration_error,
    latency_summary,
)


def _torch():
    try:
        import torch
        from torch import nn
        from torch.utils.data import DataLoader, TensorDataset
    except ImportError as error:
        raise RuntimeError("LiDAR training requires requirements-ml.txt") from error
    return torch, nn, DataLoader, TensorDataset


def build_model(scan_size=360, traversability_bins=72):
    torch, nn, _, _ = _torch()

    class LidarMultiTaskNet(nn.Module):
        def __init__(self):
            super().__init__()
            self.encoder = nn.Sequential(
                nn.Conv1d(1, 16, 7, stride=2, padding=3),
                nn.ReLU(),
                nn.Conv1d(16, 32, 5, stride=2, padding=2),
                nn.ReLU(),
                nn.Conv1d(32, 64, 5, stride=2, padding=2),
                nn.ReLU(),
                nn.AdaptiveAvgPool1d(16),
                nn.Flatten(),
            )
            self.risk = nn.Linear(64 * 16, len(RISK_LABELS))
            self.traversability = nn.Sequential(
                nn.Linear(64 * 16, traversability_bins), nn.Sigmoid()
            )
            self.direction = nn.Sequential(nn.Linear(64 * 16, 1), nn.Tanh())
            self.uncertainty = nn.Sequential(nn.Linear(64 * 16, 1), nn.Sigmoid())

        def forward(self, scan):
            features = self.encoder(scan)
            return (
                self.risk(features),
                self.traversability(features),
                self.direction(features) * 90.0,
                self.uncertainty(features),
            )

    return LidarMultiTaskNet()


def _resample(values, size):
    return [
        values[min(round(index * (len(values) - 1) / max(size - 1, 1)), len(values) - 1)]
        for index in range(size)
    ]


def _tensors(samples, torch, scan_size=360, bins=72):
    scans = torch.tensor(
        [[normalized_scan(sample, scan_size)] for sample in samples],
        dtype=torch.float32,
    )
    risk = torch.tensor(
        [RISK_LABELS.index(sample.risk_label) for sample in samples], dtype=torch.long
    )
    traversability = torch.tensor(
        [_resample(sample.traversability, bins) for sample in samples],
        dtype=torch.float32,
    )
    direction = torch.tensor(
        [[sample.recommended_direction_deg] for sample in samples],
        dtype=torch.float32,
    )
    return scans, risk, traversability, direction


def _class_weights(samples, torch):
    counts = Counter(sample.risk_label for sample in samples)
    total = len(samples)
    return torch.tensor(
        [total / max(len(RISK_LABELS) * counts.get(label, 0), 1) for label in RISK_LABELS],
        dtype=torch.float32,
    )


def _validate_training_label_coverage(samples, *, allow_incomplete=False):
    present = {sample.risk_label for sample in samples}
    missing = [label for label in RISK_LABELS if label not in present]
    if missing and not allow_incomplete:
        raise ValueError(
            "training split is missing risk labels: "
            + ", ".join(missing)
            + "; collect representative training scenarios or use "
            "--allow-incomplete-labels only for a pipeline smoke test"
        )


def _losses(nn, weights):
    return nn.CrossEntropyLoss(weight=weights), nn.BCELoss(), nn.SmoothL1Loss()


def _batch_loss(outputs, targets, losses):
    logits, predicted_map, predicted_direction, uncertainty = outputs
    target_risk, target_map, target_direction = targets
    risk_loss, map_loss, direction_loss = losses
    probabilities = logits.softmax(dim=1)
    uncertainty_target = (1.0 - probabilities.max(dim=1).values).detach().unsqueeze(1)
    return (
        risk_loss(logits, target_risk)
        + map_loss(predicted_map, target_map)
        + 0.1 * direction_loss(predicted_direction, target_direction)
        + 0.05 * map_loss(uncertainty, uncertainty_target)
    )


def _is_better_checkpoint(validation, best_f1, best_loss):
    macro_f1 = float(validation["macro_f1"])
    loss = float(validation["loss"])
    return macro_f1 > best_f1 + 1e-6 or (
        abs(macro_f1 - best_f1) <= 1e-6 and loss < best_loss - 1e-6
    )


def _evaluate(model, tensors, losses, torch):
    scans, risk, traversability, direction = tensors
    model.eval()
    with torch.no_grad():
        outputs = model(scans)
        loss = float(_batch_loss(outputs, (risk, traversability, direction), losses))
        logits, predicted_map, predicted_direction, _ = outputs
        confidence, indexes = logits.softmax(dim=1).max(dim=1)
    labels = [RISK_LABELS[int(value)] for value in risk.tolist()]
    predictions = [RISK_LABELS[int(value)] for value in indexes.tolist()]
    report = classification_report(labels, predictions, RISK_LABELS)
    report["loss"] = loss
    report["ece"] = expected_calibration_error(labels, confidence.tolist(), predictions)
    report["traversability_iou"] = sum(
        binary_iou(label, prediction)
        for label, prediction in zip(traversability.tolist(), predicted_map.tolist())
    ) / max(len(labels), 1)
    report["direction_mae_deg"] = sum(
        abs(float(actual[0]) - float(predicted[0]))
        for actual, predicted in zip(direction.tolist(), predicted_direction.tolist())
    ) / max(len(labels), 1)
    return report


def _export_and_verify(model, path, example, torch, *, model_id):
    torch.onnx.export(
        model,
        example,
        path,
        input_names=["laser_scan"],
        output_names=["risk_logits", "traversability", "direction_deg", "uncertainty"],
        dynamic_shapes={"scan": {0: torch.export.Dim("batch")}},
        opset_version=18,
    )
    try:
        import numpy as np
        import onnx
        import onnxruntime as ort
    except ImportError as error:
        raise RuntimeError("ONNX export verification requires requirements-ml.txt") from error
    model_proto = onnx.load(path)
    onnx.helper.set_model_props(
        model_proto, {"model_id": model_id, "schema_version": "1"}
    )
    onnx.save(model_proto, path)
    with torch.no_grad():
        expected = [value.detach().numpy() for value in model(example)]
    session = ort.InferenceSession(str(path), providers=["CPUExecutionProvider"])
    input_values = {session.get_inputs()[0].name: example.numpy()}
    actual = session.run(None, input_values)
    maximum_error = max(
        float(np.max(np.abs(left - right))) for left, right in zip(expected, actual)
    )
    if maximum_error > 1e-4:
        raise RuntimeError(f"PyTorch/ONNX mismatch: max error {maximum_error}")
    latencies = []
    for _ in range(30):
        started = time.perf_counter()
        session.run(None, input_values)
        latencies.append((time.perf_counter() - started) * 1000.0)
    return {
        "max_absolute_error": maximum_error,
        "cpu_latency": latency_summary(latencies),
        "providers": session.get_providers(),
    }


def train(
    dataset_path: Path,
    output_path: Path,
    *,
    epochs=20,
    batch_size=64,
    learning_rate=1e-3,
    seed=7,
    patience=5,
    model_id="lidar-risk-cnn",
    allow_incomplete_labels=False,
):
    torch, nn, DataLoader, TensorDataset = _torch()
    torch.manual_seed(seed)
    try:
        torch.use_deterministic_algorithms(True)
    except (AttributeError, RuntimeError):
        pass
    validate_dataset_directory(Path(dataset_path).parent)
    samples = load_dataset(dataset_path)
    split_samples = {
        split: [sample for sample in samples if sample.split == split]
        for split in ("train", "validation", "test")
    }
    missing = [split for split, values in split_samples.items() if not values]
    if missing:
        raise ValueError("dataset is missing required splits: " + ", ".join(missing))
    _validate_training_label_coverage(
        split_samples["train"], allow_incomplete=allow_incomplete_labels
    )
    tensors = {split: _tensors(values, torch) for split, values in split_samples.items()}
    loader = DataLoader(
        TensorDataset(*tensors["train"]),
        batch_size=batch_size,
        shuffle=True,
        generator=torch.Generator().manual_seed(seed),
    )
    model = build_model()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    losses = _losses(nn, _class_weights(split_samples["train"], torch))
    output_path = Path(output_path)
    package_dir = output_path.parent if output_path.suffix == ".onnx" else output_path
    onnx_path = output_path if output_path.suffix == ".onnx" else package_dir / "model.onnx"
    checkpoint = package_dir / "checkpoints/best.pt"
    checkpoint.parent.mkdir(parents=True, exist_ok=True)
    history = []
    best_f1, best_loss, stale_epochs = float("-inf"), float("inf"), 0
    for epoch in range(1, epochs + 1):
        model.train()
        training_loss = 0.0
        for batch_scan, batch_risk, batch_map, batch_direction in loader:
            optimizer.zero_grad()
            loss = _batch_loss(
                model(batch_scan), (batch_risk, batch_map, batch_direction), losses
            )
            loss.backward()
            optimizer.step()
            training_loss += float(loss.detach())
        validation = _evaluate(model, tensors["validation"], losses, torch)
        history.append(
            {
                "epoch": epoch,
                "train_loss": training_loss / max(len(loader), 1),
                "validation_loss": validation["loss"],
                "validation_macro_f1": validation["macro_f1"],
            }
        )
        if _is_better_checkpoint(validation, best_f1, best_loss):
            best_f1 = float(validation["macro_f1"])
            best_loss, stale_epochs = float(validation["loss"]), 0
            torch.save(model.state_dict(), checkpoint)
        else:
            stale_epochs += 1
            if stale_epochs >= patience:
                break
    model.load_state_dict(torch.load(checkpoint, weights_only=True))
    offline_metrics = {
        "validation": _evaluate(model, tensors["validation"], losses, torch),
        "test": _evaluate(model, tensors["test"], losses, torch),
    }
    package_dir.mkdir(parents=True, exist_ok=True)
    onnx_verification = _export_and_verify(
        model, onnx_path, tensors["validation"][0][:1], torch, model_id=model_id
    )
    offline_metrics["onnx"] = onnx_verification
    history_path = write_json(package_dir / "training_history.json", history)
    metrics_path = write_json(package_dir / "offline_metrics.json", offline_metrics)
    return {
        "model": onnx_path,
        "checkpoint": checkpoint,
        "history": history_path,
        "metrics": metrics_path,
        "epochs_completed": len(history),
        "seed": seed,
        "learning_rate": learning_rate,
        "batch_size": batch_size,
    }

"""Optional PyTorch trainer for the lightweight multi-task 1D CNN."""

from __future__ import annotations

import json
from pathlib import Path

from src.ml import RISK_LABELS
from src.ml.dataset import load_dataset, normalized_scan


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


def train(
    dataset_path: Path,
    output_path: Path,
    *,
    epochs=20,
    batch_size=64,
    learning_rate=1e-3,
    seed=7,
):
    torch, nn, DataLoader, TensorDataset = _torch()
    torch.manual_seed(seed)
    samples = load_dataset(dataset_path)
    training = [sample for sample in samples if sample.split == "train"]
    if not training:
        raise ValueError("dataset has no training samples")
    scan_size, bins = 360, 72
    scans = torch.tensor(
        [[normalized_scan(sample, scan_size)] for sample in training],
        dtype=torch.float32,
    )
    risk = torch.tensor(
        [RISK_LABELS.index(sample.risk_label) for sample in training], dtype=torch.long
    )
    traversability = torch.tensor(
        [
            [
                sample.traversability[
                    min(
                        round(index * (len(sample.traversability) - 1) / (bins - 1)),
                        len(sample.traversability) - 1,
                    )
                ]
                for index in range(bins)
            ]
            for sample in training
        ],
        dtype=torch.float32,
    )
    direction = torch.zeros((len(training), 1), dtype=torch.float32)
    loader = DataLoader(
        TensorDataset(scans, risk, traversability, direction),
        batch_size=batch_size,
        shuffle=True,
    )
    model = build_model(scan_size, bins)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    risk_loss = nn.CrossEntropyLoss()
    map_loss = nn.BCELoss()
    direction_loss = nn.SmoothL1Loss()
    model.train()
    history = []
    for epoch in range(epochs):
        total = 0.0
        for batch_scan, batch_risk, batch_map, batch_direction in loader:
            optimizer.zero_grad()
            logits, predicted_map, predicted_direction, uncertainty = model(batch_scan)
            loss = (
                risk_loss(logits, batch_risk)
                + map_loss(predicted_map, batch_map)
                + 0.1 * direction_loss(predicted_direction, batch_direction)
                + 0.01 * uncertainty.mean()
            )
            loss.backward()
            optimizer.step()
            total += float(loss.detach())
        history.append({"epoch": epoch + 1, "loss": total / max(len(loader), 1)})
    model.eval()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    example = torch.zeros((1, 1, scan_size), dtype=torch.float32)
    torch.onnx.export(
        model,
        example,
        output_path,
        input_names=["laser_scan"],
        output_names=[
            "risk_logits",
            "traversability",
            "direction_deg",
            "uncertainty",
        ],
        dynamic_axes={"laser_scan": {0: "batch"}},
        opset_version=17,
    )
    history_path = output_path.with_suffix(".training.json")
    history_path.write_text(json.dumps(history, indent=2) + "\n")
    return {"model": output_path, "history": history_path, "epochs": epochs}

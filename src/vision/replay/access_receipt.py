"""Resolve single-model and paired held-out access receipts."""

from __future__ import annotations

import json
from pathlib import Path


def load_replay_access_receipt(heldout_root, package):
    identity_root = Path(heldout_root) / "identity"
    single_path = identity_root / "heldout_access_receipt.json"
    if single_path.is_file():
        receipt = json.loads(single_path.read_text())
        if receipt.get("model_package_identity_sha256") != package[
            "package_identity_sha256"
        ]:
            raise ValueError("held-out receipt references a different model package")
        return receipt
    paired_path = identity_root / "paired_heldout_access_receipt.json"
    if not paired_path.is_file():
        raise ValueError("static replay requires a held-out access receipt")
    paired = json.loads(paired_path.read_text())
    matches = [
        candidate
        for candidate in paired.get("candidates", {}).values()
        if candidate.get("package_identity_sha256")
        == package["package_identity_sha256"]
    ]
    if len(matches) != 1:
        raise ValueError("paired receipt does not uniquely reference this package")
    return {
        "heldout_access_identity_sha256": paired[
            "paired_heldout_access_identity_sha256"
        ],
        "membership_sha256": paired["membership_sha256"],
        "frozen_confidence_threshold": matches[0][
            "frozen_confidence_threshold"
        ],
    }

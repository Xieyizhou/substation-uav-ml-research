"""Headless Matplotlib configuration shared by file-producing plot modules."""

from __future__ import annotations

import os
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
MPLCONFIG_DIR = PROJECT_ROOT / "outputs" / ".matplotlib_cache"


def configure_matplotlib():
    """Select a deterministic non-interactive backend before importing pyplot."""
    MPLCONFIG_DIR.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("MPLCONFIGDIR", str(MPLCONFIG_DIR))
    os.environ.setdefault("MPLBACKEND", "Agg")

    import matplotlib

    matplotlib.use("Agg", force=True)

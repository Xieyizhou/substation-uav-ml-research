"""Append versioned research samples without loading a complete dataset."""

from __future__ import annotations

import json
from pathlib import Path

from src.ml.dataset import ResearchSample


class ResearchDatasetWriter:
    def __init__(self, path: Path):
        self.path = Path(path)
        self._handle = None

    def __enter__(self):
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._handle = self.path.open("a", encoding="utf-8")
        return self

    def append(self, sample: ResearchSample):
        if self._handle is None:
            raise RuntimeError("ResearchDatasetWriter must be used as a context manager")
        sample.validate()
        self._handle.write(json.dumps(sample.to_record(), separators=(",", ":")) + "\n")
        self._handle.flush()

    def __exit__(self, exc_type, exc_value, traceback):
        if self._handle is not None:
            self._handle.close()
        self._handle = None

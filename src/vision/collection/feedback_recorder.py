"""Bounded, resumable lossless image evidence for online model feedback.

Predictions are review suggestions, never ground-truth training labels. Image
encoding runs in a bounded worker so disk latency need not block inference.
"""

import asyncio
from collections import Counter
from dataclasses import asdict, dataclass, replace
import fcntl
import hashlib
from io import BytesIO
import json
import math
from pathlib import Path
import time

import numpy as np
from PIL import Image

from src.ml.artifacts import file_sha256, object_sha256, write_json
from src.sensors.camera_decoded import decoded_content_sha256
from src.sensors.types import CameraFrame


@dataclass(frozen=True)
class FeedbackPolicy:
    max_frames: int = 256
    max_sample_bytes: int = 128 * 1024 * 1024
    minimum_interval_s: float = 0.5
    maximum_receive_age_s: float = 1.0
    max_pixels: int = 1920 * 1080
    queue_capacity: int = 2

    def __post_init__(self):
        for name in ("max_frames", "max_sample_bytes", "max_pixels", "queue_capacity"):
            if type(getattr(self, name)) is not int or getattr(self, name) <= 0:
                raise ValueError(f"{name} must be a positive integer")
        if self.max_frames > 10000 or self.queue_capacity > 8:
            raise ValueError("feedback policy exceeds bounded frame or queue limits")
        for name in ("minimum_interval_s", "maximum_receive_age_s"):
            value = getattr(self, name)
            if not math.isfinite(value) or value <= 0:
                raise ValueError(f"{name} must be finite and positive")


class FeedbackRecorder:
    def __init__(self, root, source_identity, policy=None, *, resume=False):
        self.root = Path(root).resolve()
        self.policy = policy or FeedbackPolicy()
        if self.root.exists() and not resume:
            raise ValueError("feedback output exists; use explicit resume")
        self.root.mkdir(parents=True, exist_ok=True)
        self.lock = (self.root / ".writer.lock").open("a+")
        try:
            fcntl.flock(self.lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as error:
            self.lock.close()
            raise RuntimeError("feedback recorder already running") from error
        self.counts = Counter()
        self.members = {}
        self.bytes = 0
        self.last_capture = {}
        self.closed = False
        self.config = {"feedback_schema_version": 1, "source_identity": source_identity,
                       "policy": asdict(self.policy), "training_admitted": False}
        try:
            self._recover(resume)
        except BaseException:
            self.lock.close()
            raise

    def _recover(self, resume):
        for name in ("frames", "samples"):
            directory = self.root / name
            if directory.is_symlink():
                raise ValueError("feedback directories cannot be symlinks")
        config_path = self.root / "collection.json"
        if config_path.exists():
            if json.loads(config_path.read_text()) != self.config:
                raise ValueError("feedback source or policy changed during resume")
        else:
            if resume and any(self.root.glob("samples/*.json")):
                raise ValueError("feedback configuration missing")
            write_json(config_path, self.config)
        for path in sorted(self.root.glob("samples/*.json")):
            row = json.loads(path.read_text())
            identity = row.pop("sample_identity_sha256", None)
            if identity != object_sha256(row):
                raise ValueError("feedback sample identity changed")
            frame = CameraFrame.from_record(row["frame"])
            image = (self.root / frame.payload_relative_path).resolve()
            if not image.is_relative_to(self.root) or file_sha256(image) != frame.payload_sha256:
                raise ValueError("feedback image identity changed")
            digest = row["rgb_sha256"]
            if (path.name != f"{digest}.json" or digest in self.members
                    or frame.payload_relative_path != f"frames/{digest}.png"):
                raise ValueError("feedback sample membership is invalid")
            self.members[digest] = identity
            key = (frame.source_id, frame.capture_clock_domain)
            self.last_capture[key] = max(self.last_capture.get(key, -math.inf), frame.capture_timestamp)
        self.counts["recovered_samples"] = len(self.members)
        # Interrupted writes are not admitted, but still consume disk budget.
        self.bytes = sum(path.stat().st_size for name in ("frames", "samples")
                         for path in (self.root / name).glob("*") if path.is_file())
        if len(self.members) > self.policy.max_frames or self.bytes > self.policy.max_sample_bytes:
            raise ValueError("recovered feedback exceeds its declared budget")

    def reject(self, reason):
        self.counts[reason] += 1
        return {"saved": False, "reason": reason}

    def record(self, event, predictions, *, now=None):
        if self.closed:
            raise RuntimeError("feedback recorder is closed")
        self.counts["considered"] += 1
        frame = event.frame
        now = time.monotonic() if now is None else now
        age = now - frame.receive_monotonic_timestamp
        if not math.isfinite(age) or not 0 <= age <= self.policy.maximum_receive_age_s:
            return self.reject("stale_or_future_receive_time")
        if frame.width * frame.height > self.policy.max_pixels:
            return self.reject("pixel_budget")
        digest = event.decoded.image.decoded_content_sha256
        if digest in self.members:
            return self.reject("duplicate_pixels")
        if len(self.members) >= self.policy.max_frames:
            return self.reject("frame_budget")
        key = (frame.source_id, frame.capture_clock_domain)
        if frame.capture_timestamp - self.last_capture.get(key, -math.inf) < self.policy.minimum_interval_s:
            return self.reject("sample_interval")
        rgb = event.decoded.image.array
        if (rgb.dtype != np.uint8 or rgb.shape != (frame.height, frame.width, 3)
                or decoded_content_sha256(rgb) != digest):
            raise ValueError("feedback pixels do not match decoded frame identity")
        # Encoding is lossless; labels remain unreviewed regardless of confidence.
        buffer = BytesIO()
        Image.fromarray(rgb).save(buffer, format="PNG", compress_level=1)
        payload = buffer.getvalue()
        payload_hash = hashlib.sha256(payload).hexdigest()
        saved = replace(frame, payload_format="png", pixel_format="rgb8",
                        payload_relative_path=f"frames/{digest}.png", payload_sha256=payload_hash)
        row = {"frame": saved.to_record(), "rgb_sha256": digest,
               "original_payload_sha256": frame.payload_sha256,
               "predictions": predictions, "annotation_status": "unreviewed",
               "training_admitted": False, "source_identity": self.config["source_identity"]}
        row["sample_identity_sha256"] = object_sha256(row)
        encoded = (json.dumps(row, sort_keys=True, indent=2, ensure_ascii=False, allow_nan=False) + "\n").encode()
        if len(encoded) > 128 * 1024:
            return self.reject("metadata_budget")
        size = len(payload) + len(encoded)
        image_path = self.root / saved.payload_relative_path
        if image_path.is_symlink():
            raise ValueError("feedback output cannot be a symlink")
        replaced_bytes = image_path.stat().st_size if image_path.exists() else 0
        if self.bytes + size - replaced_bytes > self.policy.max_sample_bytes:
            return self.reject("storage_budget")
        image_path.parent.mkdir(exist_ok=True)
        temporary = image_path.with_suffix(".png.partial")
        temporary.write_bytes(payload)
        temporary.replace(image_path)
        # Publish metadata last: only committed samples participate in resume.
        metadata = self.root / "samples" / f"{digest}.json"
        write_json(metadata, row)
        self.members[digest] = row["sample_identity_sha256"]
        self.bytes += size - replaced_bytes
        self.last_capture[key] = frame.capture_timestamp
        self.counts["saved"] += 1
        return {"saved": True, "sample_identity_sha256": row["sample_identity_sha256"],
                "frame": saved.to_record(), "bytes": size}

    def close(self, *, error=None):
        if self.closed:
            return self.summary
        self.summary = {**self.config, "state": "failed" if error else "complete",
                        "error": error, "session_counts": dict(self.counts),
                        "sample_count": len(self.members), "sample_bytes": self.bytes,
                        "membership_sha256": object_sha256(sorted(self.members.items()))}
        self.summary["collection_identity_sha256"] = object_sha256(self.summary)
        try:
            write_json(self.root / "summary.json", self.summary)
        finally:
            self.closed = True
            self.lock.close()
        return self.summary


class BoundedFeedbackWriter:
    def __init__(self, recorder):
        self.recorder = recorder
        self.queue = asyncio.Queue(maxsize=recorder.policy.queue_capacity)
        self.error = None
        self.closed = False
        self.task = asyncio.create_task(self._run())

    def submit(self, event, predictions):
        if self.closed:
            raise RuntimeError("feedback writer is closed")
        if self.error is not None:
            return self.recorder.reject("writer_failed")
        if self.queue.full():
            self.queue.get_nowait()
            self.recorder.reject("queue_replaced")
        self.queue.put_nowait((event, predictions))
        return {"queued": True}

    async def _run(self):
        while True:
            item = await self.queue.get()
            if item is None:
                return
            if self.error is not None:
                self.recorder.reject("writer_failed")
                continue
            try:
                await asyncio.to_thread(self.recorder.record, *item)
            except Exception as error:
                self.error = error

    async def close(self, *, error=None):
        self.closed = True
        await self.queue.put(None)
        await self.task
        summary = self.recorder.close(error=str(self.error) if self.error else error)
        if self.error:
            raise RuntimeError("feedback persistence failed") from self.error
        return summary

"""Verified YOLO inference feeding synchronized, bounded visual route requests."""

import asyncio
from collections import deque
import math
from pathlib import Path
import time

import numpy as np

from src.flight.aligned_sim_rgbd import AlignedSimRgbdSource
from src.flight.visual_target_route import StableVisualTarget
from src.ml.artifacts import write_json, file_sha256
from src.sandbox.visual_runtime_model import visual_runtime_model
from src.sensors.gazebo_depth_memory import aligned_depth, bbox_depth
from src.vision.collection.feedback_recorder import FeedbackPolicy, FeedbackRecorder, BoundedFeedbackWriter
from src.vision.localization.rgbd_localization import localize_bbox


class VisualReplanService:
    def __init__(self, model_run, output):
        self.info = visual_runtime_model(model_run)
        self.output = Path(output)
        self.target = StableVisualTarget()
        self.recent = deque(maxlen=6)
        self.ready = asyncio.Event()
        self.task = self.source = self.writer = None
        self.active = False
        self.processed = 0
        self.error = None
        self.stopped = False

    @property
    def returncode(self):
        return 1 if self.task is not None and self.task.done() else None

    async def start(self, drone):
        from ultralytics import YOLO
        self.output.mkdir(parents=True, exist_ok=False)
        self.model = YOLO(str(self.info["model_path"]), task="detect")
        write_json(self.output / "model.json", {key: str(value) if isinstance(value, Path) else value for key, value in self.info.items()})
        await asyncio.to_thread(self.model.predict, source=np.zeros((self.info["imgsz"], self.info["imgsz"], 3), np.uint8),
                                imgsz=self.info["imgsz"], rect=False, device="cpu", verbose=False)
        self.source = AlignedSimRgbdSource(drone, gazebo_sitl=True)
        await self.source.start()
        self.task = asyncio.create_task(self._run())

    def begin_mission(self):
        self.target = StableVisualTarget()
        self.recent.clear()
        self.writer = BoundedFeedbackWriter(FeedbackRecorder(self.output / "feedback", {
            "model_sha256": self.info["model_sha256"], "receipt": self.info["receipt_identity_sha256"]},
            FeedbackPolicy(max_frames=64, max_sample_bytes=16*1024**2)))
        self.active = True

    def _predict(self, event):
        result = self.model.predict(source=np.ascontiguousarray(event.decoded.image.array[:, :, ::-1]),
                                    imgsz=self.info["imgsz"], conf=max(.85, self.info["threshold"]), rect=False,
                                    device="cpu", iou=.7, verbose=False)[0]
        return [] if result.boxes is None else [dict(class_name=result.names[int(cls)], confidence=conf, bbox=box)
            for cls, conf, box in zip(result.boxes.cls.tolist(), result.boxes.conf.tolist(), result.boxes.xyxy.tolist())]

    async def _run(self):
        try:
            async for event, depth, match in self.source.events():
                predictions = await asyncio.to_thread(self._predict, event)
                now = time.monotonic()
                try:
                    aligned_depth(event.frame, depth, now=now)
                    match = self.source.pose.match(event.frame, now=now)
                except ValueError as exc:
                    self.source.counts["rejected_after_inference: "+str(exc)] += 1
                    self.target = StableVisualTarget()
                    continue
                intrinsics = dict(fx=event.frame.width/(2*math.tan(1.466/2)),
                                  fy=event.frame.width/(2*math.tan(1.466/2)), cx=event.frame.width/2, cy=event.frame.height/2)
                observations = []
                for row in predictions:
                    if min(row["bbox"][0], row["bbox"][1], event.frame.width-row["bbox"][2], event.frame.height-row["bbox"][3]) < 2:
                        continue
                    distance = bbox_depth(depth, (event.frame.width, event.frame.height), row["bbox"])
                    if distance is not None:
                        position = localize_bbox(distance, row["bbox"], intrinsics, match["pose"],
                                                 dict(forward_m=.18, right_m=0, down_m=-.12))
                        horizontal = math.hypot(position["east_m"]-match["pose"]["east_m"],
                                                position["north_m"]-match["pose"]["north_m"])
                        # A five-metre stand-off must be reachable within the
                        # existing bounded flight region. Do not pursue far
                        # high-confidence equipment outside that approach range.
                        if not 5.5 <= horizontal <= 10:
                            self.source.counts["target_outside_approach_range"] += 1
                            continue
                        observations.append({**row, "depth_m": distance, "local_position": position})
                evidence = dict(capture_timestamp=event.frame.capture_timestamp,
                                rgb_sha256=event.decoded.image.decoded_content_sha256,
                                depth_sha256=depth.payload_sha256, depth_timestamp=depth.capture_timestamp,
                                model_sha256=self.info["model_sha256"], pose_match=match,
                                observations=observations)
                self.processed += 1
                self.ready.set()
                if self.active:
                    self.recent.append((event, depth, evidence))
                    self.target.update(observations, evidence, now=now)
                    self.writer.submit(event, predictions)
                write_json(self.output / "latest.json", dict(evidence=evidence, target=self.target.current(now),
                                                            processed=self.processed, control_authority="bounded_semantic_request"))
        except BaseException as exc:
            if not isinstance(exc, asyncio.CancelledError):
                self.error = f"{type(exc).__name__}: {exc}"
            raise

    def current(self, now):
        if self.task is None or self.task.done():
            raise RuntimeError(self.error or "visual service not running")
        self.source.check_health()
        return self.target.current(now)

    async def snapshot(self, request):
        wanted = {row["capture_timestamp"] for row in request["evidence"]}
        members = [item for item in self.recent if item[0].frame.capture_timestamp in wanted]
        if len(members) != 3:
            raise ValueError("visual request frames expired from buffer")
        root = self.output / "accepted-request"
        def save():
            recorder = FeedbackRecorder(root, {"request_identity": request["identity"]},
                                        FeedbackPolicy(max_frames=3, max_sample_bytes=8*1024**2, minimum_interval_s=.001))
            error = None
            try:
                for event, depth, evidence in members:
                    result = recorder.record(event, evidence["observations"], now=event.frame.receive_monotonic_timestamp)
                    if not result["saved"] and result["reason"] != "duplicate_pixels":
                        raise ValueError("could not preserve accepted visual frame")
                    path = root / (depth.payload_sha256 + ".npz")
                    np.savez_compressed(path, depth=depth.array)
                paths = list(root.rglob("*.png")) + list(root.glob("*.npz"))
                write_json(root / "request.json", dict(request=request, inputs={str(p): file_sha256(p) for p in paths}))
            except BaseException as exc:
                error = str(exc)
                raise
            finally:
                recorder.close(error=error)
        await asyncio.to_thread(save)
        return str(root / "request.json")

    async def stop(self):
        if self.stopped:
            return
        self.stopped = True
        if self.task:
            self.task.cancel()
            await asyncio.gather(self.task, return_exceptions=True)
        try:
            if self.source:
                await self.source.stop()
        finally:
            if self.writer:
                await self.writer.close(error=self.error)
            if self.output.exists():
                write_json(self.output / "receipt.json", dict(error=self.error, processed=self.processed,
                           counts=dict(self.source.counts) if self.source else {}, model_sha256=self.info["model_sha256"],
                           control_authority="bounded_semantic_request", training_admitted=False))

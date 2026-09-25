"""Collect bounded PNG review samples from live YOLO or historical sidecar replay."""

import argparse
import asyncio
import hashlib
import json
from pathlib import Path
import time
from types import SimpleNamespace
from zipfile import ZipFile

import numpy as np

from src.ml.artifacts import file_sha256, object_sha256, write_json
from src.sensors.camera_decoded import decoded_content_sha256
from src.sensors.camera_decoder import decode_camera_payload
from src.sensors.types import CameraFrame
from src.vision.collection.feedback_recorder import FeedbackPolicy, FeedbackRecorder, BoundedFeedbackWriter


def _source_record(root):
    path = root / "completion.json"
    value = json.loads(path.read_text())
    identity = value.pop("identity", None)
    if (identity != object_sha256(value) or value.get("training_admitted") is not False
            or value.get("status") != "live_complete_not_flight_certified"
            or value.get("control_authority") != "none"):
        raise ValueError("source must have an intact non-training sidecar receipt")
    if file_sha256(root / "detections.jsonl") != value["inputs"][str(root / "detections.jsonl")]:
        raise ValueError("source detection timeline changed")
    return file_sha256(path)


def replay(recording, output, policy, *, limit=300, resume=False):
    root = Path(recording).resolve()
    identity = _source_record(root)
    archive = None
    if (root / "raw-evidence.zip").is_file():
        index = json.loads((root / "raw-evidence-archive.json").read_text())
        supplied = index.pop("archive_identity_sha256", None)
        if (supplied != object_sha256(index) or index["completion_sha256"] != identity
                or file_sha256(root / "raw-evidence.zip") != index["archive_sha256"]):
            raise ValueError("source camera archive changed")
        archive = ZipFile(root / "raw-evidence.zip")
    recorder = FeedbackRecorder(output, {"source_receipt_sha256": identity}, policy, resume=resume)
    error = None
    try:
        with (root / "detections.jsonl").open() as stream:
            for index, line in enumerate(stream):
                if index >= limit:
                    break
                row = json.loads(line)
                frame = CameraFrame.from_record(row["frame"])
                if frame.width * frame.height > policy.max_pixels:
                    recorder.reject("pixel_budget")
                    continue
                if frame.payload_format == "raw" and archive is not None:
                    member = f"payloads/{frame.payload_relative_path}"
                    if frame.pixel_format != "rgb8" or archive.getinfo(member).file_size != frame.width * frame.height * 3:
                        raise ValueError("archived frame is not packed RGB8")
                    payload = archive.read(member)
                    if hashlib.sha256(payload).hexdigest() != frame.payload_sha256:
                        raise ValueError("source payload identity changed")
                    rgb = np.frombuffer(payload, np.uint8).reshape(frame.height, frame.width, 3)
                    decoded = SimpleNamespace(image=SimpleNamespace(array=rgb,
                        decoded_content_sha256=decoded_content_sha256(rgb)))
                else:
                    decoded = decode_camera_payload(frame, root / "payloads")
                if decoded.image.decoded_content_sha256 != row["rgb_sha256"]:
                    raise ValueError("source decoded pixels changed")
                event = SimpleNamespace(frame=frame, decoded=decoded)
                recorder.record(event, row["predictions"], now=row["result_monotonic"])
    except BaseException as exc:
        error = f"{type(exc).__name__}: {exc}"
        raise
    finally:
        summary = recorder.close(error=error)
        if archive is not None:
            archive.close()
    return summary


async def observe(model, model_info, output, policy, *, topic, seconds, source_factory=None):
    from src.sensors.gazebo_camera_latest import GazeboLatestMemorySource

    source_factory = source_factory or GazeboLatestMemorySource
    root = Path(output)
    recorder = FeedbackRecorder(root / "feedback", {
        "model_sha256": model_info["onnx_model_sha256"],
        "model_receipt_sha256": model_info["receipt_identity_sha256"],
        "topic": topic,
    }, policy)
    writer = BoundedFeedbackWriter(recorder)
    source = source_factory(root / "unused-memory-source", topic=topic, encoder_workers=1)
    queue = asyncio.Queue(maxsize=1)
    counts = {"received": 0, "replaced_before_decode": 0, "invalid": 0,
              "stale": 0, "inferred": 0}

    async def receive():
        async for event in source.events(timeout_s=5):
            counts["received"] += 1
            if queue.full():
                queue.get_nowait()
                counts["replaced_before_decode"] += 1
            queue.put_nowait(event)

    def predict(event):
        result = model.predict(source=np.ascontiguousarray(event.decoded.image.array[:, :, ::-1]),
                               imgsz=model_info["imgsz"], conf=0.05, iou=0.7, max_det=300,
                               rect=False, device="cpu", verbose=False)[0]
        if result.boxes is None:
            return []
        return [{"class_name": result.names[int(cls)], "confidence": float(conf), "xyxy": box}
                for cls, conf, box in zip(result.boxes.cls.tolist(), result.boxes.conf.tolist(), result.boxes.xyxy.tolist())]

    task, error = None, None
    try:
        await source.start()
        task = asyncio.create_task(receive())
        deadline = time.monotonic() + seconds
        while time.monotonic() < deadline:
            if task.done():
                await task
                raise RuntimeError("camera source ended")
            try:
                raw = await asyncio.wait_for(queue.get(), min(1.0, max(0.001, deadline-time.monotonic())))
            except asyncio.TimeoutError:
                continue
            if not 0 <= time.monotonic() - raw.received <= policy.maximum_receive_age_s:
                counts["stale"] += 1
                continue
            try:
                event = await asyncio.to_thread(raw.materialize)
            except (ValueError, TypeError):
                counts["invalid"] += 1
                continue
            predictions = await asyncio.to_thread(predict, event)
            finished = time.monotonic()
            counts["inferred"] += 1
            writer.submit(event, predictions)
            write_json(root / "latest.json", {
                "frame": event.frame.to_record(), "predictions": predictions,
                "result_monotonic": finished,
                "receive_to_result_s": finished-event.frame.receive_monotonic_timestamp,
                "model_sha256": model_info["onnx_model_sha256"],
                "threshold": model_info["threshold"], "control_authority": "none",
                "planning_eligible": False,
                "planning_rejection": "pose_depth_and_capture_clock_alignment_required",
                "payload_retention": "only_samples_committed_to_feedback_are_persisted",
                "training_admitted": False,
            })
        if not counts["inferred"]:
            raise RuntimeError("no valid camera inference")
        if task.done():
            await task
            raise RuntimeError("camera source ended")
    except BaseException as exc:
        error = f"{type(exc).__name__}: {exc}"
        raise
    finally:
        if task is not None:
            task.cancel()
            await asyncio.gather(task, return_exceptions=True)
        try:
            await source.stop()
        except BaseException as exc:
            error = error or f"{type(exc).__name__}: {exc}"
            raise
        finally:
            try:
                summary = await writer.close(error=error)
            except BaseException as exc:
                error = error or f"{type(exc).__name__}: {exc}"
                raise
            finally:
                write_json(root / "status.json", {"state": "failed" if error else "complete",
                    "error": error, "counts": counts, "control_authority": "none",
                    "training_admitted": False})
    return {"counts": counts, "feedback": summary}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("replay", "live"), required=True)
    parser.add_argument("--recording", type=Path)
    parser.add_argument("--model-run", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--max-frames", type=int, default=256)
    parser.add_argument("--max-mib", type=int, default=128)
    parser.add_argument("--interval", type=float, default=0.5)
    parser.add_argument("--limit", type=int, default=300)
    parser.add_argument("--seconds", type=float, default=30)
    parser.add_argument("--topic", default="/research_camera/image")
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    policy = FeedbackPolicy(max_frames=args.max_frames, max_sample_bytes=args.max_mib*1024**2,
                            minimum_interval_s=args.interval)
    if args.mode == "replay":
        if args.recording is None or not 0 < args.limit <= 100000:
            parser.error("replay requires --recording and a bounded positive --limit")
        result = replay(args.recording, args.output, policy, limit=args.limit, resume=args.resume)
    else:
        if args.model_run is None or not 0 < args.seconds <= 240 or args.resume:
            parser.error("live requires --model-run, 0 < seconds <= 240, and a new output")
        from ultralytics import YOLO
        from src.sandbox.workbench_inference import verified_model
        from scripts.vision.locked_cpu_threads import locked_threads
        info = verified_model(args.model_run)
        with locked_threads(4):
            model = YOLO(str(info["onnx_path"]))
            model.predict(source=np.zeros((info["imgsz"], info["imgsz"], 3), dtype=np.uint8),
                          imgsz=info["imgsz"], device="cpu", rect=False, verbose=False)
            result = asyncio.run(observe(model, info, args.output, policy, topic=args.topic, seconds=args.seconds))
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()

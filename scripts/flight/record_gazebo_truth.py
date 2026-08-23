#!/usr/bin/env python3
"""Write-only normalized Gazebo truth recorder; never imported by runtime."""
import argparse, asyncio, hashlib, json, sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.vision.collection.gazebo_truth import GazeboTruthSource


async def record(args):
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.receipt.parent.mkdir(parents=True, exist_ok=True)
    source = GazeboTruthSource(width=args.width, height=args.height, topic=args.topic)
    count = 0
    invalid_count = 0
    digest = hashlib.sha256()
    error = None
    try:
        await source.start()
        with args.output.open("w", encoding="utf-8") as target:
            async for event in source.events(timeout_s=args.timeout):
                row = event.truth.to_record()
                encoded = json.dumps(row, sort_keys=True, separators=(",", ":"))
                target.write(encoded + "\n")
                digest.update((encoded + "\n").encode())
                count += 1
                invalid_count += not event.truth.valid
                if args.frames and count >= args.frames:
                    break
    except (RuntimeError, TimeoutError, ValueError) as exception:
        error = f"{type(exception).__name__}: {exception}"
    finally:
        await source.stop()
    complete = count > 0 and invalid_count == 0 and (not args.frames or count == args.frames)
    receipt = {
        "schema_version": 2,
        "evidence_role": "offline_truth_only",
        "status": "complete" if complete else "blocked",
        "topic": source.topic,
        "image_width": args.width,
        "image_height": args.height,
        "requested_frames": args.frames,
        "message_count": count,
        "invalid_message_count": invalid_count,
        "stream_sha256": digest.hexdigest(),
        "error": error,
    }
    receipt["identity"] = hashlib.sha256(json.dumps(receipt, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
    args.receipt.write_text(json.dumps(receipt, indent=2, sort_keys=True) + "\n")
    return 0 if complete else 2


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--topic", default="auto")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--receipt", type=Path, required=True)
    parser.add_argument("--frames", type=int, default=0)
    parser.add_argument("--timeout", type=float, default=15.0)
    parser.add_argument("--width", type=int, default=1920)
    parser.add_argument("--height", type=int, default=1080)
    args = parser.parse_args()
    if args.frames < 0 or args.timeout <= 0 or args.width <= 0 or args.height <= 0:
        parser.error("frames must be non-negative and dimensions/timeout positive")
    return asyncio.run(record(args))


if __name__ == "__main__":
    raise SystemExit(main())

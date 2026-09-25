"""Verify bounded feedback capture against the existing static Gazebo scene."""

import argparse
import asyncio
import json
import os
from pathlib import Path
import signal
import sys

from scripts.vision.prepare_paired_visual_factors import BASE
from scripts.vision.probe_material_shadow import command
from src.ml.artifacts import file_sha256, write_json
from src.vision.canonical.plan import CARRIER, read_record


async def probe(output, model_run):
    output = Path(output).resolve()
    output.mkdir(parents=True, exist_ok=False)
    base = BASE / "original/plan"
    plan = read_record(base / "plan.json")
    world = base / "world.sdf"
    if file_sha256(world) != plan["files"]["world.sdf"]:
        raise ValueError("probe scene identity changed")
    os.environ.update(GZ_PARTITION=f"feedback_probe_{os.getpid()}", GZ_IP="127.0.0.1")
    server = child = None
    error = None
    try:
        with (output / "gazebo.log").open("x") as log:
            server = await asyncio.create_subprocess_exec("gz", "sim", "-s", "-r", str(world),
                stdout=log, stderr=asyncio.subprocess.STDOUT, start_new_session=True)
            for _ in range(40):
                if server.returncode is not None:
                    raise RuntimeError("Gazebo exited")
                if "/research_camera/image" in (await command("gz", "topic", "-l")).splitlines():
                    break
                await asyncio.sleep(1)
            else:
                raise TimeoutError("camera topic unavailable")
            view = plan["calibration_views"][0]
            x, y, z = view["position"]
            qx, qy, qz, qw = view["orientation"]
            request = f'name: "{CARRIER}" position {{x:{x} y:{y} z:{z}}} orientation {{x:{qx} y:{qy} z:{qz} w:{qw}}}'
            response = await command("gz", "service", "-s", f'/world/{plan["world_name"]}/set_pose',
                "--reqtype", "gz.msgs.Pose", "--reptype", "gz.msgs.Boolean", "--timeout", "3000", "--req", request)
            if "true" not in response:
                raise RuntimeError("pose request rejected")
            with (output / "collector.log").open("x") as childlog:
                child = await asyncio.create_subprocess_exec(sys.executable, "-m", "scripts.vision.collect_feedback",
                    "--mode", "live", "--model-run", str(Path(model_run).resolve()), "--seconds", "12",
                    "--max-frames", "16", "--max-mib", "8", "--output", str(output / "live"),
                    stdout=childlog, stderr=asyncio.subprocess.STDOUT, start_new_session=True)
                await asyncio.wait_for(child.wait(), 75)
                if child.returncode != 0:
                    raise RuntimeError("feedback collector failed; see collector.log")
            status = json.loads((output / "live/status.json").read_text())
            summary = json.loads((output / "live/feedback/summary.json").read_text())
            if status["state"] != "complete" or not 0 < summary["sample_count"] <= 16 or summary["sample_bytes"] > 8*1024**2:
                raise RuntimeError("live collection did not satisfy sample or storage limits")
            if list((output / "live").rglob("*.raw")):
                raise RuntimeError("unbounded raw payload files were written")
    except BaseException as exc:
        error = f"{type(exc).__name__}: {exc}"
        raise
    finally:
        for process in (child, server):
            if process is not None and process.returncode is None:
                os.killpg(process.pid, signal.SIGINT)
                try:
                    await asyncio.wait_for(process.wait(), 8)
                except asyncio.TimeoutError:
                    os.killpg(process.pid, signal.SIGKILL)
                    await process.wait()
        proof = {"state": "failed" if error else "complete", "error": error,
                 "owned_processes_exited": all(p is None or p.returncode is not None for p in (server, child)),
                 "world_sha256": file_sha256(world), "px4_started": False, "flight_tested": False,
                 "inputs": {str(path.resolve()): file_sha256(path) for path in
                            (Path(__file__), Path("scripts/vision/collect_feedback.py"),
                             Path("src/vision/collection/feedback_recorder.py"))}}
        write_json(output / "probe.json", proof)
    return proof


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--model-run", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(asyncio.run(probe(args.output, args.model_run)), indent=2))

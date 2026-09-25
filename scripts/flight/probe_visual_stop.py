"""Request controlled stop only after measured resumed visual-route motion."""

import argparse
import asyncio
import json
import math
import os
from pathlib import Path
import signal
import sys
import time

from src.ml.artifacts import file_sha256, write_json
from src.sandbox.live_replan_gate import BASE
from src.sandbox.live_replan_stop import validate_run_id


async def probe(run_id, output):
    validate_run_id(run_id)
    output = Path(output).resolve()
    output.mkdir(parents=True, exist_ok=False)
    run = Path(BASE).resolve() / run_id
    if run.exists():
        raise FileExistsError(run)
    request, process, error = None, None, None
    logfile = output / "flight.log"
    try:
        with logfile.open("x") as log:
            process = await asyncio.create_subprocess_exec(sys.executable, "-u", "-m", "scripts.flight.fly_visual_replan",
                "--fly", "--run-id", run_id, stdout=log, stderr=asyncio.subprocess.STDOUT, start_new_session=True)
            deadline = time.monotonic()+480
            while process.returncode is None and time.monotonic() < deadline:
                if request is None and "route_replaced" in logfile.read_text():
                    telemetry = run / "runtime/telemetry.jsonl"
                    with telemetry.open("rb") as stream:
                        stream.seek(0, 2)
                        size = stream.tell()
                        stream.seek(max(0, size-16384))
                        lines = stream.read().decode(errors="replace").splitlines()
                    for line in reversed(lines):
                        try:
                            row = json.loads(line)
                        except ValueError:
                            continue
                        if row.get("stream") != "local":
                            continue
                        speed = math.hypot(row["value"]["vn"], row["value"]["ve"])
                        if row["phase"] == "post_hover_trial" and speed >= .1 and time.monotonic()-row["monotonic"] < .5:
                            request = dict(reason="operator stop probe after actual resumed motion", speed_m_s=speed,
                                           monotonic=time.monotonic(), simulation_only=True)
                            write_json(run / "stop-requested.json", request)
                            print("CONTROLLED_STOP_REQUESTED", flush=True)
                        break
                await asyncio.sleep(.1)
            if process.returncode is None:
                write_json(run / "stop-requested.json", dict(reason="probe timeout", simulation_only=True))
                await asyncio.wait_for(process.wait(), 90)
                raise TimeoutError("visual stop probe exceeded its observation budget")
            if request is None:
                raise RuntimeError("flight ended before a moving stop request")
            runtime = json.loads((run / "runtime/receipt.json").read_text())
            route = json.loads((run / "route.json").read_text())
            if not (process.returncode != 0 and runtime["landing_confirmed"] and runtime["final_armed"] is False
                    and runtime["owned_processes_exited"] and "Sandbox requested controlled stop and landing" in runtime["error"]
                    and route["status"] == "failed"):
                raise RuntimeError("stop did not produce confirmed controlled landing and cleanup")
    except BaseException as exc:
        error = f"{type(exc).__name__}: {exc}"
        raise
    finally:
        if process is not None and process.returncode is None:
            write_json(run / "stop-requested.json", dict(reason="probe cleanup", simulation_only=True))
            try:
                await asyncio.wait_for(process.wait(), 90)
            except asyncio.TimeoutError:
                os.killpg(process.pid, signal.SIGINT)
                await asyncio.wait_for(process.wait(), 15)
        paths = [Path(__file__).resolve(), logfile, run / "protocol.json", run / "route.json",
                 run / "runtime/receipt.json", run / "stop-requested.json"]
        result = dict(state="failed" if error else "controlled_visual_stop_verified", error=error,
                      run=str(run), request=request, process_exited=process is not None and process.returncode is not None,
                      inputs={str(path): file_sha256(path) for path in paths if path.is_file()})
        write_json(output / "proof.json", result)
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(asyncio.run(probe(args.run_id, args.output)), indent=2))

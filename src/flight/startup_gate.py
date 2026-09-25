"""Unarmed startup ordering. No flight envelope or failsafe is relaxed."""
import asyncio
from pathlib import Path
import time

def startup_state(text):
    if 'Startup script returned with return value:' in text:raise RuntimeError('PX4 startup script failed; inspect startup trace')
    return 'Startup script returned successfully' in text

async def wait_px4_startup(log,processes,timeout_s=90.,check_cancel=None):
    if not 0<timeout_s<=90:raise ValueError('Startup wait must be bounded by 90 seconds')
    start=time.monotonic()
    while True:
        if check_cancel is not None:check_cancel()
        if any(p.returncode is not None for p in processes):raise RuntimeError('Simulator process exited before startup gate')
        if startup_state(Path(log).read_text()):return time.monotonic()-start
        if time.monotonic()-start>=timeout_s:raise TimeoutError('PX4 startup completion gate timed out; inspect startup trace')
        await asyncio.sleep(.2)

"""Bounded asyncio shutdown helpers for MAVSDK flight processes."""

import asyncio
import contextlib
import signal
import threading


async def cancel_tasks(tasks, timeout_s):
    """Cancel tasks and return any that ignored cancellation before the timeout."""
    active = [task for task in tasks if task is not None and not task.done()]
    for task in active:
        task.cancel()
    if not active:
        return []
    done, pending = await asyncio.wait(active, timeout=max(float(timeout_s), 0.01))
    for task in done:
        with contextlib.suppress(asyncio.CancelledError, Exception):
            task.exception()
    return list(pending)


def run_with_bounded_shutdown(coroutine, shutdown_timeout_s):
    """Run a coroutine without allowing cancellation-resistant tasks to hang exit."""
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    root_task = loop.create_task(coroutine)
    previous_handlers = {}
    stop_requested = False
    def request_stop(signum, frame):
        nonlocal stop_requested
        if not stop_requested:
            stop_requested = True
            loop.call_soon_threadsafe(root_task.cancel)
    if threading.current_thread() is threading.main_thread():
        for signum in (signal.SIGINT, signal.SIGTERM):
            previous_handlers[signum] = signal.signal(signum, request_stop)
    try:
        return loop.run_until_complete(root_task)
    finally:
        for signum, handler in previous_handlers.items():
            signal.signal(signum, handler)
        pending = list(asyncio.all_tasks(loop))
        if pending:
            remaining = loop.run_until_complete(
                cancel_tasks(pending, shutdown_timeout_s)
            )
            for task in remaining:
                task._log_destroy_pending = False
        try:
            loop.run_until_complete(
                asyncio.wait_for(
                    loop.shutdown_asyncgens(),
                    timeout=max(float(shutdown_timeout_s), 0.01),
                )
            )
        except (asyncio.CancelledError, Exception):
            pass
        asyncio.set_event_loop(None)
        loop.close()

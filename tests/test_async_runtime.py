import asyncio
import unittest

from src.flight.async_runtime import cancel_tasks, run_with_bounded_shutdown


class AsyncCancellationTests(unittest.IsolatedAsyncioTestCase):
    async def test_normal_task_cancels_within_timeout(self):
        task = asyncio.create_task(asyncio.sleep(60))
        pending = await cancel_tasks([task], 0.1)
        self.assertEqual(pending, [])
        self.assertTrue(task.cancelled())


class BoundedLoopShutdownTests(unittest.TestCase):
    def test_cancellation_resistant_task_does_not_block_process_exit(self):
        async def resistant_task():
            try:
                await asyncio.Event().wait()
            except asyncio.CancelledError:
                await asyncio.Event().wait()

        async def main_task():
            asyncio.create_task(resistant_task(), name="resistant-test-task")
            await asyncio.sleep(0)
            return "finished"

        result = run_with_bounded_shutdown(main_task(), 0.01)
        self.assertEqual(result, "finished")


if __name__ == "__main__":
    unittest.main()

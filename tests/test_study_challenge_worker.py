import unittest
from unittest.mock import patch

from src.study.challenge_worker import execute_challenge


class ChallengeWorkerTests(unittest.TestCase):
    @patch("src.study.challenge_worker.execute_flight_tier")
    def test_executes_only_the_challenge_tier(self, execute):
        execute.return_value = {"completed": 3}
        result = execute_challenge("registry.sqlite", "study-1", "outputs")
        self.assertEqual(result["completed"], 3)
        self.assertEqual(execute.call_args.kwargs["tier"], "challenge")


if __name__ == "__main__":
    unittest.main()

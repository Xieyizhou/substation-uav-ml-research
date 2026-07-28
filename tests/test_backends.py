import asyncio
import unittest

from src.backends.base import FlightBackend
from src.backends.dji_psdk_backend import DjiPsdkBackend


class BackendContractTests(unittest.TestCase):
    def test_dji_backend_implements_high_level_contract(self):
        backend = DjiPsdkBackend()
        self.assertIsInstance(backend, FlightBackend)
        self.assertEqual(backend.backend_id, "dji_psdk")
        self.assertFalse(backend.health().connected)

    def test_dji_commands_require_connected_bridge(self):
        backend = DjiPsdkBackend()
        with self.assertRaisesRegex(RuntimeError, "not connected"):
            asyncio.run(backend.hover())


if __name__ == "__main__":
    unittest.main()

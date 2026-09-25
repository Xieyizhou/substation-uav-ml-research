import ast
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch
import unittest
from scripts.flight import check_px4_shadow as check

class Px4ShadowTests(unittest.TestCase):
    def test_refuses_existing_px4(self):
        with patch.object(check.subprocess,'run',return_value=SimpleNamespace(returncode=0)),self.assertRaisesRegex(RuntimeError,'Existing PX4'):check.preflight()

    def test_no_arming_or_motion_api(self):
        tree=ast.parse(Path(check.__file__).read_text())
        calls=[n.func.attr for n in ast.walk(tree) if isinstance(n,ast.Call) and isinstance(n.func,ast.Attribute)]
        for name in ('arm','takeoff','land','goto_location','set_position_ned','set_velocity_ned','set_actuator_control','upload_mission','start_mission'):
            self.assertNotIn(name,calls)
        text=Path(check.__file__).read_text()
        self.assertIn('udpin://127.0.0.1:14547',text)
        self.assertIn("str(BUILD/'etc')",text)

if __name__=='__main__':unittest.main()

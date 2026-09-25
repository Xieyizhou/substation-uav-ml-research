"""The documented Demo launcher works with an actual package-free Python."""

import json
import os
from pathlib import Path
import shutil
import subprocess
import tempfile
import unittest
import venv


class SandboxLauncherTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.temporary = tempfile.TemporaryDirectory()
        cls.addClassCleanup(cls.temporary.cleanup)
        cls.root = Path(cls.temporary.name).resolve()
        venv.create(cls.root / 'python', with_pip=False)
        cls.python = cls.root / 'python/bin/python'
        (cls.root / 'scripts').mkdir()
        source = Path(__file__).resolve().parents[1] / 'scripts/run_sandbox_app.sh'
        shutil.copy2(source, cls.root / 'scripts/run_sandbox_app.sh')
        (cls.root / 'main.py').write_text(
            'import json,sys\n'
            'print(json.dumps({"python":sys.executable,"args":sys.argv[1:]}))\n'
        )
        result = subprocess.run(
            [str(cls.python), '-c', 'import importlib.util; assert importlib.util.find_spec("mavsdk") is None'],
            capture_output=True, text=True,
        )
        if result.returncode:
            raise AssertionError(result.stderr)

    def launch(self, explicit):
        env = dict(os.environ)
        env.pop('UAV_SANDBOX_PYTHON', None)
        env.pop('PYTHONPATH', None)
        env['PATH'] = f'{self.python.parent}:/usr/bin:/bin'
        if explicit:
            env['UAV_SANDBOX_PYTHON'] = str(self.python)
        result = subprocess.run(
            [str(self.root / 'scripts/run_sandbox_app.sh'), '--port', '8877'],
            cwd=self.root, env=env, capture_output=True, text=True, timeout=15,
        )
        self.assertEqual(result.returncode, 0, result.stderr)
        output = json.loads(result.stdout)
        self.assertEqual(Path(output['python']).parent.resolve(), self.python.parent.resolve())
        self.assertEqual(output['args'], [
            'sandbox', '--project-root', str(self.root.resolve()),
            '--profile', 'demo', 'serve', '--port', '8877',
        ])

    def test_demo_accepts_package_free_explicit_python(self):
        self.launch(explicit=True)

    def test_demo_discovers_python_on_path(self):
        self.launch(explicit=False)

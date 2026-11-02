"""Exercise real collection, failures and zero-test refusal in child processes."""
import json
from pathlib import Path
import subprocess
import sys

import pytest

from scripts.run_tests import modules_for

ROOT = Path(__file__).resolve().parents[1]


def execute(tmp_path, files, *, require_all=True):
    (tmp_path / "tests").mkdir()
    for name, content in files.items():
        (tmp_path / "tests" / (name + ".py")).write_text(content)
    report = tmp_path / "report.json"
    program = (
        "import json,sys;"
        f"sys.path.insert(0,{str(ROOT)!r});"
        "from scripts.run_tests import run_modules;"
        f"r=run_modules({list(files)!r},root={str(tmp_path)!r},require_all={require_all!r});"
        f"open({str(report)!r},'w').write(json.dumps(r));"
        "sys.exit(0 if r['passed'] else 1)"
    )
    process = subprocess.run([sys.executable, "-c", program], capture_output=True, text=True, timeout=30)
    assert report.is_file(), process.stdout + process.stderr
    return process.returncode, json.loads(report.read_text())


def test_unittest_functions_and_parameterized_cases_are_all_executed(tmp_path):
    code, result = execute(tmp_path, {"test_mixed": (
        "import unittest,pytest\n"
        "class Case(unittest.TestCase):\n"
        " def test_method(self): self.assertTrue(True)\n"
        "def test_function(): assert True\n"
        "@pytest.mark.parametrize('value',[1,2])\n"
        "def test_parameter(value): assert value > 0\n"
    )})
    assert code == 0 and result["passed"]
    assert result["tests"] == 4
    assert result["collected_by_module"] == {"tests/test_mixed.py": 4}


def test_an_empty_module_cannot_hide_behind_another_nonempty_module(tmp_path):
    code, result = execute(tmp_path, {
        "test_empty": "# No executable tests\n",
        "test_present": "def test_present(): assert True\n",
    })
    assert code != 0 and not result["passed"]
    assert result["missing_modules"] == ["tests/test_empty.py"]


@pytest.mark.parametrize("content,field", [
    ("def test_bad(): assert False\n", "failures"),
    ("import missing_publication_fixture_dependency\n", "errors"),
])
def test_failures_and_import_errors_fail_the_run(tmp_path, content, field):
    code, result = execute(tmp_path, {"test_broken": content})
    assert code != 0 and not result["passed"] and result[field] > 0


def test_skips_cannot_qualify_the_portable_suite(tmp_path):
    code, result = execute(tmp_path, {"test_skip": (
        "import pytest\n"
        "@pytest.mark.skip(reason='required dependency missing')\n"
        "def test_skipped(): pass\n"
    )})
    assert code != 0 and not result["passed"] and result["skipped"]


def test_manifest_groups_are_explicit_and_cannot_overlap(tmp_path):
    (tmp_path / "tests").mkdir()
    for name in ("portable", "native", "sitl", "research"):
        (tmp_path / "tests" / ("test_" + name + ".py")).write_text("def test_ok(): pass\n")
    path = tmp_path / "config/sandbox/test_suites.json"
    path.parent.mkdir(parents=True)
    record = dict(research_modules=["test_research"], native_modules=["test_native"], sitl_modules=["test_sitl"])
    path.write_text(json.dumps(record))
    assert modules_for("core", tmp_path) == ["test_portable"]
    assert modules_for("native", tmp_path) == ["test_native"]
    assert modules_for("sitl", tmp_path) == ["test_sitl"]
    record["sitl_modules"].append("test_native")
    path.write_text(json.dumps(record))
    with pytest.raises(ValueError):
        modules_for("core", tmp_path)

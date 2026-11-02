"""Run portable tests with pytest or explicitly selected, isolated research tests."""
from __future__ import annotations

import argparse
from collections import Counter
import json
import os
from pathlib import Path
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
GROUPS = {"research": "research_modules", "native": "native_modules", "sitl": "sitl_modules"}


def modules_for(suite, root=ROOT):
    root = Path(root)
    manifest = json.loads((root / "config/sandbox/test_suites.json").read_text())
    names = sorted(p.stem for p in (root / "tests").glob("test_*.py"))
    selected = {}
    seen = set()
    for group, field in GROUPS.items():
        values = manifest.get(field, [])
        if (not isinstance(values, list) or len(set(values)) != len(values)
                or set(values) - set(names) or seen.intersection(values)):
            raise ValueError("Test manifest has duplicate, overlapping or missing modules")
        selected[group] = set(values)
        seen.update(values)
    if suite == "all":
        return names
    if suite == "core":
        return [name for name in names if name not in seen]
    if suite not in GROUPS:
        raise ValueError("Unknown test suite")
    return [name for name in names if name in selected[suite]]


class TestAudit:
    def __init__(self, paths):
        self.paths = {str(path.resolve()) for path in paths}
        self.items = {}
        self.collected_by_module = Counter()
        self.failed = set()
        self.errors = set()
        self.skipped = {}
        self.collection_errors = 0
        self.missing = []

    def pytest_collection_finish(self, session):
        for item in session.items:
            self.items[item.nodeid] = str(item.path.resolve())
            self.collected_by_module[str(item.path.resolve())] += 1
        self.missing = sorted(self.paths - set(self.collected_by_module))

    def pytest_collectreport(self, report):
        if report.failed:
            self.collection_errors += 1

    def pytest_runtest_logreport(self, report):
        if report.failed:
            if report.when == "call":
                self.failed.add(report.nodeid)
            else:
                self.errors.add(report.nodeid)
        if report.skipped:
            self.skipped[report.nodeid] = str(report.longrepr)

    def pytest_sessionfinish(self, session, exitstatus):
        # A nonempty second module cannot hide an empty first module.
        if self.missing and exitstatus == 0:
            session.exitstatus = 5


def run_modules(names, *, require_all=False, root=ROOT):
    root = Path(root).resolve()
    paths = [root / "tests" / (name + ".py") for name in names]
    if not paths:
        return dict(passed=False, tests=0, failures=0, errors=0, skipped=[],
                    exit_code=5, missing_modules=[], runner="pytest")
    try:
        import pytest
    except ImportError:
        return dict(passed=False, tests=0, failures=0, errors=1, skipped=[],
                    exit_code=4, missing_modules=[],
                    error="Install requirements-test.txt in this Python environment.",
                    runner="pytest")
    sys.path.insert(0, str(root))
    sys.path.insert(0, str(root / "tests"))
    audit = TestAudit(paths)
    previous = os.environ.get("PYTEST_DISABLE_PLUGIN_AUTOLOAD")
    os.environ["PYTEST_DISABLE_PLUGIN_AUTOLOAD"] = "1"
    try:
        code = int(pytest.main([
            "-q", "-p", "no:cacheprovider", "--rootdir", str(root),
            *(str(path) for path in paths),
        ], plugins=[audit]))
    finally:
        if previous is None:
            os.environ.pop("PYTEST_DISABLE_PLUGIN_AUTOLOAD", None)
        else:
            os.environ["PYTEST_DISABLE_PLUGIN_AUTOLOAD"] = previous
    skipped = [{"test": name, "reason": reason} for name, reason in sorted(audit.skipped.items())]
    return dict(
        passed=code == 0 and bool(audit.items) and not audit.missing
               and not (require_all and skipped),
        tests=len(audit.items), failures=len(audit.failed),
        errors=len(audit.errors) + audit.collection_errors,
        skipped=skipped, exit_code=code, runner="pytest",
        missing_modules=[str(Path(path).relative_to(root)) for path in audit.missing],
        collected_by_module={
            str(Path(path).relative_to(root)): count
            for path, count in sorted(audit.collected_by_module.items())
        },
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--suite", choices=("core", "research", "native", "sitl", "all"), default="core")
    parser.add_argument("--list", action="store_true")
    parser.add_argument("--report", type=Path)
    parser.add_argument("--module", help=argparse.SUPPRESS)
    args = parser.parse_args()
    try:
        names = modules_for(args.suite)
    except ValueError as error:
        parser.error(str(error))
    if args.module:
        if args.module not in names:
            parser.error("Module is not in the selected suite")
        names = [args.module]
    if args.list:
        print("\n".join(names))
        return 0
    started = time.monotonic()
    if args.module or args.suite == "core":
        result = run_modules(names, require_all=True)
    else:
        # Frozen adapters configure module globals and remain process-isolated.
        rows = []
        for name in names:
            print(f"Checking {name}", flush=True)
            proc = subprocess.run([
                sys.executable, str(Path(__file__).resolve()),
                "--suite", args.suite, "--module", name,
            ], cwd=ROOT)
            rows.append(dict(module=name, exit_code=proc.returncode))
        result = dict(passed=bool(rows) and all(row["exit_code"] == 0 for row in rows),
                      modules=rows, runner="pytest", process_isolated=True)
    report = dict(suite=args.suite, module_count=len(names), **result,
                  elapsed_seconds=round(time.monotonic() - started, 3))
    if args.report:
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))
    return 0 if result["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())

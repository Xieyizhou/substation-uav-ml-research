"""Run portable core checks or explicitly selected historical research checks."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import subprocess
import sys
import time
import unittest

ROOT = Path(__file__).resolve().parents[1]


def modules_for(suite, root=ROOT):
    manifest = json.loads((root / "config/sandbox/test_suites.json").read_text())
    research = manifest["research_modules"]
    names = sorted(p.stem for p in (root / "tests").glob("test_*.py"))
    if len(set(research)) != len(research) or set(research) - set(names):
        raise ValueError("Research test manifest has duplicate or missing modules")
    # New checks are core by default; they cannot silently disappear from CI.
    return [name for name in names if suite == "all" or
            (name in research) == (suite == "research")]


def run_modules(names, *, require_all=False):
    sys.path.insert(0, str(ROOT))
    sys.path.insert(0, str(ROOT / "tests"))
    suite = unittest.TestSuite(unittest.defaultTestLoader.loadTestsFromName(name) for name in names)
    result = unittest.TextTestRunner(verbosity=1).run(suite)
    return dict(passed=result.wasSuccessful() and not (require_all and result.skipped), tests=result.testsRun,
                failures=len(result.failures), errors=len(result.errors),
                skipped=[{"test": str(test), "reason": reason} for test, reason in result.skipped])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--suite", choices=("core", "research", "all"), default="core")
    parser.add_argument("--list", action="store_true")
    parser.add_argument("--report", type=Path)
    parser.add_argument("--module", help=argparse.SUPPRESS)
    args = parser.parse_args()
    names = modules_for(args.suite)
    if args.module:
        if args.module not in names:
            parser.error("Module is not in the selected suite")
        names = [args.module]
    if args.list:
        print("\n".join(names))
        return 0
    started = time.monotonic()
    if args.module or args.suite == "core":
        result = run_modules(names, require_all=args.suite != "core")
    else:
        # Frozen adapters intentionally configure module globals. Process
        # isolation preserves those old source hashes and their original usage.
        results = []
        for name in names:
            print(f"Checking {name}", flush=True)
            proc = subprocess.run([sys.executable, str(Path(__file__).resolve()),
                                   "--suite", args.suite, "--module", name], cwd=ROOT)
            results.append(dict(module=name, exit_code=proc.returncode))
        result = dict(passed=all(row["exit_code"] == 0 for row in results), modules=results)
    report = dict(suite=args.suite, module_count=len(names), **result,
                  elapsed_seconds=round(time.monotonic() - started, 3))
    if args.report:
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))
    return 0 if result["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())

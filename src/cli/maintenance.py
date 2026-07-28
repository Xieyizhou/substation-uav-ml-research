"""Infrequent project data-maintenance commands."""

from __future__ import annotations

import argparse
import sys

from src.cli.process import forwarded_args, run_module


def main(argv=None):
    parser = argparse.ArgumentParser(
        prog="python main.py maintenance",
        description="Infrequent project data-maintenance tools.",
    )
    actions = parser.add_subparsers(dest="action")
    actions.add_parser(
        "migrate-outputs",
        help="Move legacy outputs/as_* folders into stage folders",
    )
    argv = list(sys.argv[1:] if argv is None else argv)
    if not argv or argv[0] in {"-h", "--help"}:
        parser.print_help()
        return 0 if argv else 2
    if argv[0] != "migrate-outputs":
        print(f"Unknown maintenance action: {argv[0]}", file=sys.stderr)
        parser.print_usage(sys.stderr)
        return 2
    return run_module(
        "src.logging.migrate_outputs_to_stages",
        forwarded_args(argv[1:]),
        "Migrating legacy output folders",
    )

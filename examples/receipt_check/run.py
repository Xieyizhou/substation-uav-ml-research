"""Run the receipt check offline example."""
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from src.sandbox.offline_examples import example_main

if __name__ == "__main__":
    raise SystemExit(example_main("receipt_check"))

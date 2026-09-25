"""Additional review-gate regressions and runtime start evidence for scale control."""
import argparse
import subprocess
import sys
from pathlib import Path
from unittest.mock import patch
from scripts.vision import routed_scale_control as arm
from src.ml.artifacts import file_sha256
from src.vision.canonical.plan import write_record

def run(train=False):
    arm.configure();arm.runtime.TESTS=tuple(dict.fromkeys(arm.runtime.TESTS+('tests.test_routed_scale_gate',)))
    arm.runtime.launch_gate(create=True)
    if train:arm.runtime.train()

if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--train',action='store_true');a=ap.parse_args();run(a.train)

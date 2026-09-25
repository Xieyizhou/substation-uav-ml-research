"""Explicit continuation to error-review handoff; no automatic visual decisions."""
import argparse
from scripts.vision.train_reviewed_hold import preflight,run
from scripts.vision.reviewed_hold_gate import check
from scripts.vision.evaluate_reviewed_hold import evaluate,evidence


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--run',action='store_true');args=ap.parse_args()
    preflight();check()
    if not args.run:print('PREFLIGHT_ONLY_NO_TRAINING');return
    run();evaluate();evidence()
    print('EXPLICIT_ERROR_REVIEW_REQUIRED; NO_AUTOMATIC_PASS')


if __name__=='__main__':main()

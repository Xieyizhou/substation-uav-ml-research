"""Evaluate each completed tranche without interrupting the fixed training grid."""
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from scripts.vision.exposure_protocol import OUT


def main():
    last_count = -1
    while True:
        count = len(list(OUT.glob('*/completion.json')))
        if count != last_count and (count % 3 == 0 or count == 18):
            print(f'EVALUATING_TRANCHE {count}/18', flush=True)
            subprocess.run([sys.executable, str(ROOT/'scripts/vision/evaluate_exposure_diagnosis.py')], check=True)
            last_count = count
        if count == 18:
            subprocess.run([sys.executable, str(ROOT/'scripts/vision/finalize_exposure_diagnosis.py')], check=True)
            subprocess.run([sys.executable, str(ROOT/'scripts/vision/report_exposure_diagnosis.py')], check=True)
            return
        time.sleep(30)


if __name__ == '__main__':
    main()

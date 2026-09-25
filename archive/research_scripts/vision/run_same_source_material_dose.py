"""Explicit staged runner. Defaults to preflight; never writes review decisions."""
import argparse,subprocess,sys
from scripts.vision import preflight_same_source_material_dose as preflight
from scripts.vision import finalize_same_source_material_ready as gate
from scripts.vision import train_same_source_material_dose as training

def main():
    p=argparse.ArgumentParser();p.add_argument('--execute',action='store_true');a=p.parse_args()
    preflight.run();gate.main()
    if not a.execute:print('READY_NO_TRAINING');return
    training.run()
    subprocess.run([sys.executable,'-u','-m','scripts.vision.evaluate_same_source_material_dose','--evaluate'],check=True,cwd=training.prior.ROOT)
    print('NUMERICAL_COMPLETE_EXPLICIT_REVIEW_REQUIRED',flush=True)

if __name__=='__main__':main()

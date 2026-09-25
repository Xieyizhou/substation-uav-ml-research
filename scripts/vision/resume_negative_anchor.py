"""Resume frozen runner with explicit seed order, independent of JSON key order."""
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
import scripts.vision.run_negative_anchor as runner

def ordered(protocol):
    protocol=dict(protocol)
    protocol['schedules']={f'H-100-{s}':protocol['schedules'][f'H-100-{s}'] for s in runner.SEEDS}
    return protocol

if __name__=='__main__':
    prepare=runner.prepare
    runner.prepare=lambda:ordered(prepare())
    runner.main()

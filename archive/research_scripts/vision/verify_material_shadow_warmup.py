"""Confirm warmup preserves all 96 development predictions and matches."""
from pathlib import Path
from unittest.mock import patch
from scripts.vision import verify_material_shadow_fix as check
from scripts.vision.material_shadow import warmup

def main():
    original=check.YOLO
    def warmed(*a,**kw):
        model=original(*a,**kw);warmup(model);return model
    out=Path('data/research/material-shadow-v1/prewarm-offline-v1').resolve()
    with patch.object(check,'OUT',out),patch.object(check,'YOLO',warmed):check.main()

if __name__=='__main__':main()

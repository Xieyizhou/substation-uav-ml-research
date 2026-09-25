"""Use unchanged isolated probe for three prewarmed runs."""
import asyncio
from pathlib import Path
from unittest.mock import patch
from scripts.vision import retest_material_shadow as repeat

ROOT=Path('data/research/material-shadow-v1/prewarm-retest-v1').resolve()

if __name__=='__main__':
    with patch.object(repeat,'ROOT',ROOT):asyncio.run(repeat.main())

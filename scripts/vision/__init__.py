"""Current vision commands, with import compatibility for research archives.

Desktop and main CLI dependencies remain here. Historical modules live under
archive/research_scripts/vision; the fallback preserves their Python imports
and ``python -m scripts.vision.<name>`` entry points without wrapper copies.
"""

from pathlib import Path

_archive = Path(__file__).resolve().parents[2] / "archive/research_scripts/vision"
if _archive.is_dir():
    __path__.append(str(_archive))

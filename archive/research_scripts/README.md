# Historical research scripts

These scripts are retained for research provenance and historical experiments.
The desktop application, main CLI, packaging commands, and the source files
required by current flight evidence remain in `src/` and `scripts/`.

`manifest.json` records each original path, archive path, and unchanged SHA-256.
No dataset, model, or experiment receipt was deleted or rewritten during this
move. This archive separates historical work; it is not a disk-space cleanup.

Python imports and `python -m scripts.vision.<module>` continue to resolve through
the package search path. Direct filesystem paths have changed. Historical
experiments that hash their original script paths or derive paths from
`__file__` may require restoring their files before reproducing that experiment.
Do not rewrite historical receipt hashes to approve a moved script.

To restore a particular archived source, copy its `archived` manifest path to
its `original` path after verifying its SHA-256. Existing original files take
precedence over the archive in Python's module search. Restore related files
together when replaying a historical protocol. Current flight launch gates do
not depend on moved files and are checked separately after refactoring.

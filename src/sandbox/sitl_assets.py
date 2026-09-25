"""Materialize only fixed-scene assets from a pinned portable evidence archive."""

import json
from pathlib import Path
import shutil
import tempfile

from src.ml.artifacts import file_sha256
from src.sandbox.evidence_archive import EvidenceArchive
from src.sandbox.evidence_registry import registered_archive

ASSETS = Path('outputs/sandbox/runtime_assets/fixed-scene-v1')
# Suffixes are provenance identifiers, never original filesystem paths to open.
BASE = 'data/research/material-shadow-v1/autonomy-avoidance-v1/'
MEMBERS = {
    'world.sdf': (BASE+'envelope-aware-scene-v1/world.sdf', '033affe105a146e966aeda02dd7f38cb78806c8470ba092638df4a01c049ec63'),
    'server.config': (BASE+'magnetic-contract-001/server.config', 'dfc873d6cccdf64005d493b75c3a1f8ebb29d9ecf76fbfd5989ab5507b770828'),
    'envelope.json': (BASE+'vehicle-envelope-002/envelope.json', '9fb728bb4bff51ab1146c707fd28476dd2394fe4c21b68abbb4fd5ddf8ebfee3'),
    'obstacles.json': ('config/substation_obstacles.json', 'a707aa2147480b2244d8db3acf13aa0cf2a20425b556c87e80dbd31559092d57'),
    'models/x500_research_lowload/model.sdf': (BASE+'lowload-vehicle-v1/models/x500_research_lowload/model.sdf', 'dcedfbf5ac9f632ee6c2bc2290537e9229536ea19282ee57d35a5fd272f932d1'),
    'models/x500_research_lowload/model.config': (BASE+'lowload-vehicle-v1/models/x500_research_lowload/model.config', '711d823203f53d4c88ad10900113bcb2d2c5bc2c57839c2fe68396d4f63dfd32'),
}


def verify_assets(root):
    directory = Path(root) / ASSETS
    if directory.is_symlink() or not directory.resolve().is_relative_to(Path(root).resolve()):
        raise ValueError('Runtime assets escape project root')
    for name, (_, digest) in MEMBERS.items():
        path = directory / name
        if path.is_symlink() or not path.resolve().is_relative_to(directory.resolve()) or file_sha256(path) != digest:
            raise ValueError('Fixed runtime asset changed: '+name)
    return directory


def install_assets(root):
    root = Path(root)
    entry, archive_path = registered_archive(root)
    destination = root / ASSETS
    if destination.exists():
        return verify_assets(root)
    destination.parent.mkdir(parents=True, exist_ok=True)
    if not destination.parent.resolve().is_relative_to(root.resolve()):
        raise ValueError('Runtime asset root escapes project')
    stage = Path(tempfile.mkdtemp(prefix='.fixed-scene-', dir=destination.parent))
    try:
        with EvidenceArchive(archive_path, expected_identity=entry['manifest_sha256']) as archive:
            provenance = {}
            for name, (suffix, digest) in MEMBERS.items():
                keys = [key for key in archive.files if key == suffix or key.endswith('/'+suffix)]
                if len(keys) != 1 or archive.files[keys[0]]['sha256'] != digest:
                    raise ValueError('Fixed asset missing or ambiguous: '+name)
                target = stage/name
                target.parent.mkdir(parents=True, exist_ok=True)
                target.write_bytes(archive.read(keys[0]))
                provenance[name] = dict(sha256=digest, archive_key=keys[0])
            (stage/'origin.json').write_text(json.dumps(dict(manifest_sha256=archive.identity, members=provenance,
                scope='Scenario assets only; historical results do not qualify current runtime'), indent=2)+'\n')
        stage.rename(destination)
    finally:
        if stage.exists():
            shutil.rmtree(stage)
    return verify_assets(root)

"""Current fixed-scene SITL identity and independent positive/stop qualification."""

from datetime import datetime, timezone
from importlib.metadata import version
import json
import math
import os
from pathlib import Path
import platform
import re
import shutil
import subprocess
import sys

from src.ml.artifacts import file_sha256, object_sha256
from src.sandbox.sitl_assets import verify_assets, MEMBERS
from src.sandbox.workbench_recipe import IDENTIFIER
from src.sandbox.visual_runtime_model import visual_runtime_model

RUNS = Path('outputs/sandbox/semantic_runs')
ACTION = 'semantic-flight'
INTENTS = {'qualify-positive', 'qualify-stop', 'mission'}


def run_directory(root, run_id):
    if not isinstance(run_id, str) or re.fullmatch(r'semantic-[0-9a-f]{32}', run_id) is None:
        raise ValueError('Invalid semantic flight run ID')
    root = Path(root).resolve()
    path = root / RUNS / run_id
    if path.is_symlink() or not path.resolve().is_relative_to(root):
        raise ValueError('Semantic run escapes project root')
    return path


def model_directory(config, model_id):
    if not isinstance(model_id, str) or not IDENTIFIER.fullmatch(model_id):
        raise ValueError('Invalid selected model ID')
    path = config.workbench_runs_root / model_id
    if path.is_symlink() or not path.resolve().is_relative_to(config.workbench_runs_root.resolve()):
        raise ValueError('Selected model escapes workbench')
    return path


def current_identity(config, model_id):
    """Rehash on launch; identities use logical relative names, not machine paths."""
    root, px4 = Path(config.project_root).resolve(), Path(config.px4_root).resolve()
    assets = verify_assets(root)
    info = visual_runtime_model(model_directory(config, model_id))
    inputs = {}
    def bind(label, path):
        inputs[label] = dict(path=str(path.resolve()), sha256=file_sha256(path))
    # Conservative source closure: changing any current Python core invalidates
    # qualification. Historical experiment directories are deliberately absent.
    for path in sorted((root/'src').rglob('*.py')):
        bind('source/'+path.relative_to(root).as_posix(), path)
    for name in ('scripts/maps/prepare_research_vehicle.py', 'scripts/vision/locked_cpu_threads.py'):
        bind('source/'+name, root/name)
    for name in MEMBERS:
        bind('scenario/'+name, assets/name)
    bind('model/onnx', info['model_path'])
    bind('model/receipt', info['receipt_path'])
    build = px4/'build/px4_sitl_default'
    bind('px4/bin/px4', build/'bin/px4')
    for prefix in ('build/px4_sitl_default/etc', 'Tools/simulation/gz/models/x500', 'Tools/simulation/gz/models/x500_base'):
        paths = sorted(path for path in (px4/prefix).rglob('*') if path.is_file())
        if not paths:
            raise ValueError('PX4 runtime resources missing: '+prefix)
        for path in paths:
            bind('px4/'+path.relative_to(px4).as_posix(), path)
    plugins = sorted((build/'src/modules/simulation/gz_plugins').rglob('*.dylib')) + sorted((build/'src/modules/simulation/gz_plugins').rglob('*.so'))
    for path in plugins:
        bind('px4/'+path.relative_to(px4).as_posix(), path)
    gz = shutil.which('gz')
    if not gz:
        raise ValueError('Gazebo command is unavailable')
    bind('gazebo/command', Path(gz))
    gz_version = subprocess.run([gz, 'sim', '--versions'], capture_output=True, text=True, check=True, timeout=5).stdout.strip()
    environment = dict(gazebo_sim=gz_version, python=platform.python_version(), system=platform.platform(), architecture=platform.machine(),
        packages={name: version(name) for name in ('numpy','mavsdk','torch','ultralytics','onnxruntime','Pillow')},
        controls={key: value for key, value in sorted(os.environ.items()) if key.startswith(('PX4_PARAM_', 'PX4_GZ_')) and key not in {'PX4_GZ_MAG_ENU_GAUSS','PX4_PARAM_EKF2_DECL_TYPE','PX4_PARAM_EKF2_MAG_DECL'}},
        scope='Fixed local PX4/Gazebo SITL, not physical flight or cross-platform certification')
    identity = dict(schema_version=1, model_id=model_id, model_receipt=info['receipt_identity_sha256'],
                    files={key: row['sha256'] for key, row in inputs.items()}, environment=environment)
    return dict(identity=identity, identity_sha256=object_sha256(identity), inputs=inputs)


def _checked(path):
    record = json.loads(path.read_text())
    identity = record.pop('identity_sha256', None)
    if identity != object_sha256(record):
        raise ValueError('Qualification record identity changed')
    return record


def qualification(config, identity):
    verified, invalid = {}, []
    for path in sorted((config.project_root/RUNS).glob('*/qualification.json')):
        try:
            value = _checked(path)
            if value['runtime_identity_sha256'] != identity:
                continue
            if value['kind'] not in ('positive', 'controlled_stop'):
                raise ValueError('Unknown qualification kind')
            required = {'protocol.json','runtime/receipt.json','runtime/telemetry.jsonl','vision/receipt.json','route.json'}
            required |= {'completion.json','inair-alignment.json','prearm-alignment.json','evidence-replay.json','vision/accepted-request/request.json'} if value['kind']=='positive' else {'stop-requested.json'}
            if not required.issubset(value['files']):
                raise ValueError('Incomplete qualification evidence membership')
            for name, digest in value['files'].items():
                target = path.parent/name
                if Path(name).is_absolute() or '..' in Path(name).parts or not target.resolve().is_relative_to(path.parent.resolve()) or file_sha256(target) != digest:
                    raise ValueError('Qualification evidence changed')
            protocol = json.loads((path.parent/'protocol.json').read_text())
            if object_sha256(protocol['runtime_identity']) != identity:
                raise ValueError('Runtime identity contents changed')
            if protocol['runtime_identity_sha256'] != identity or protocol['intent'] != 'qualify-'+('positive' if value['kind']=='positive' else 'stop'):
                raise ValueError('Qualification protocol mismatch')
            runtime = json.loads((path.parent/'runtime/receipt.json').read_text())
            if not (runtime['landing_confirmed'] is True and runtime['final_armed'] is False and runtime['owned_processes_exited'] is True):
                raise ValueError('Qualification landing incomplete')
            _verify_outcome(path.parent, value['kind'])
            verified[value['kind']] = path.parent.name
        except (OSError, ValueError, KeyError, TypeError) as error:
            invalid.append(dict(run_id=path.parent.name, reason=str(error)))
    return dict(qualified=set(verified)=={'positive','controlled_stop'}, evidence=verified, invalid=invalid,
                runtime_identity_sha256=identity, simulation_only=True)


def _verify_outcome(out, kind):
    protocol = json.loads((out/'protocol.json').read_text())
    runtime = json.loads((out/'runtime/receipt.json').read_text())
    if not (runtime['landing_confirmed'] is True and runtime['final_armed'] is False and runtime['owned_processes_exited'] is True):
        raise ValueError('Landing or cleanup incomplete')
    if kind == 'controlled_stop':
        if not (out/'stop-requested.json').is_file() or runtime['status'] != 'failed' or runtime['error'] != 'RuntimeError: Sandbox requested controlled stop and landing':
            raise ValueError('Explicit controlled stop not demonstrated')
        route = json.loads((out/'route.json').read_text())
        if not any(math.hypot(row['position']['vn'], row['position']['ve']) >= .1 for row in route['samples']):
            raise ValueError('Stop must be tested after actual horizontal motion')
        if not any(row['event']=='failsafe_landing_confirmed' for row in runtime['events']):
            raise ValueError('Controlled stop landing event missing')
    elif kind == 'positive':
        done = json.loads((out/'completion.json').read_text())
        if done['status'] != 'visual_stop_replan_resume_standoff_verified':
            raise ValueError('Positive flight audit missing')
    else:
        raise ValueError('Unknown qualification kind')


def write_qualification(out, kind):
    _verify_outcome(out, kind)
    protocol = json.loads((out/'protocol.json').read_text())
    files = {path.relative_to(out).as_posix(): file_sha256(path) for path in sorted(out.rglob('*'))
             if path.is_file() and path.name != 'qualification.json'}
    value = dict(schema_version=1, kind=kind, runtime_identity_sha256=protocol['runtime_identity_sha256'],
                 verified_at=datetime.now(timezone.utc).isoformat(), files=files, simulation_only=True,
                 scope='One positive and one explicit in-motion controlled stop are required for this exact current identity')
    value['identity_sha256'] = object_sha256(value)
    with (out/'qualification.json').open('x') as stream:
        json.dump(value, stream, indent=2)
        stream.write('\n')
    return value

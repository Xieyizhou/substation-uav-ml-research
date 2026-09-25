"""Desktop status of selected-model runtime qualification and owned telemetry."""

import json
import math
import subprocess
from pathlib import Path

from src.sandbox.semantic_qualification import ACTION, RUNS, current_identity, qualification, run_directory
from src.sandbox.workbench_inference import list_verified_models


def summary(config, status):
    jobs = ([status['active_job']] if status['active_job'] else [])+status['history']
    job = next((row for row in jobs if row['action']==ACTION), None)
    result = dict(models=list_verified_models(config.workbench_runs_root), latest_run=None)
    if job is None:
        return result
    out = run_directory(config.project_root, job['scenario_id'])
    value = dict(job_id=job['job_id'], state=job['state'], run_id=out.name, phase='startup', speed_m_s=None)
    try:
        value.update({key: json.loads((out/'protocol.json').read_text())[key] for key in ('model_id','intent')})
        with (out/'runtime/telemetry.jsonl').open('rb') as stream:
            stream.seek(0,2)
            size=stream.tell();stream.seek(max(0,size-65536))
            lines=stream.read().splitlines()
        for line in reversed(lines):
            try:
                row=json.loads(line)
                if row['stream']=='local':
                    value.update(phase=row['phase'],speed_m_s=math.hypot(row['value']['vn'],row['value']['ve']))
                    break
            except (ValueError,KeyError,TypeError):
                continue
        if (out/'qualification.json').is_file():
            value['qualification_kind']=json.loads((out/'qualification.json').read_text())['kind']
        if (out/'completion.json').is_file():
            value['completion']=json.loads((out/'completion.json').read_text())['status']
    except (OSError,ValueError,KeyError,TypeError):
        pass
    result['latest_run']=value
    return result


def model_status(config, model_id):
    try:
        if config.profile != 'development':
            raise ValueError('Selected-model SITL is available only in development profile')
        # Resolve the selected model's deterministic scene cache on first use.
        # This only materializes six pinned assets; it never starts a runtime.
        from src.sandbox.sitl_assets import install_assets
        install_assets(config.project_root)
        identity=current_identity(config,model_id)
        return dict(available=True, **qualification(config,identity['identity_sha256']))
    except (OSError,ValueError,KeyError,TypeError,ImportError,subprocess.SubprocessError) as error:
        return dict(available=False,qualified=False,reason=str(error))

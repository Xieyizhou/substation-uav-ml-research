"""Desktop command boundary for selected-model fixed-scene SITL."""

import sys
from pathlib import Path
from uuid import uuid4

from src.sandbox.command_models import SandboxCommand
from src.sandbox.semantic_qualification import (ACTION, INTENTS, RUNS, current_identity,
    qualification)
from src.sandbox.sitl_assets import install_assets
from src.sandbox.workflow import WorkflowArtifact


def build_semantic_command(config, scenario_id, parameters):
    if config.profile != 'development' or scenario_id is not None:
        raise ValueError('Selected-model SITL is development-only with a fixed scene')
    if not isinstance(parameters, dict) or set(parameters) != {'model_id', 'intent'}:
        raise ValueError('Select model_id and intent')
    if parameters['intent'] not in INTENTS:
        raise ValueError('Unknown SITL intent')
    install_assets(config.project_root)
    current = current_identity(config, parameters['model_id'])
    if parameters['intent'] == 'mission' and not qualification(config, current['identity_sha256'])['qualified']:
        raise ValueError('Selected model/current runtime needs positive and controlled-stop qualification first')
    run_id = 'semantic-'+uuid4().hex
    output = (RUNS/run_id).as_posix()
    expected = ['runtime/receipt.json', 'vision/receipt.json', 'protocol.json']
    expected += ['qualification.json'] if parameters['intent'].startswith('qualify-') else ['completion.json']
    receipt = current['inputs']['model/receipt']
    return SandboxCommand(ACTION, (sys.executable, '-m', 'src.sandbox.semantic_entry',
        '--project-root', str(config.project_root), '--run-id', run_id,
        '--model-id', parameters['model_id'], '--intent', parameters['intent'],
        '--identity', current['identity_sha256']), 600., scenario_id=run_id,
        workflow='selected_model_fixed_scene_sitl',
        artifacts=(WorkflowArtifact('selected_model_receipt', receipt['sha256'],
                   str(Path(receipt['path']).relative_to(config.project_root))),),
        expected_outputs=tuple(output+'/'+name for name in expected), budget_paths=(output,))

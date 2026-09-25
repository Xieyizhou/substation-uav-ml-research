"""Integrity, isolation and cancellation boundaries of current SITL qualification."""
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from src.ml.artifacts import object_sha256
from src.sandbox.semantic_qualification import (RUNS, current_identity, model_directory,
    qualification, run_directory, write_qualification)
from src.sandbox.semantic_commands import build_semantic_command
from src.sandbox.managed_flight_stop import cooperative_stop
from src.sandbox.sitl_assets import verify_assets


class QualificationTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.config = SimpleNamespace(project_root=self.root, profile='development',
            workbench_runs_root=self.root/'models')
        self.identity = {'code':'a'*64,'model':'b'*64}
        self.digest = object_sha256(self.identity)

    def attempt(self, kind, number):
        out=run_directory(self.root,'semantic-'+f'{number:032x}')
        out.mkdir(parents=True)
        rows={
            'protocol.json':dict(intent='qualify-'+('positive' if kind=='positive' else 'stop'),runtime_identity=self.identity,runtime_identity_sha256=self.digest),
            'runtime/receipt.json':dict(status='sitl_hover_landed_disarmed' if kind=='positive' else 'failed',landing_confirmed=True,final_armed=False,owned_processes_exited=True,
                error=None if kind=='positive' else 'RuntimeError: Sandbox requested controlled stop and landing',events=[dict(event='failsafe_landing_confirmed')]),
            'route.json':dict(samples=[dict(position=dict(vn=.2,ve=0))]),
            'vision/receipt.json':{}, 'runtime/telemetry.jsonl':{},
            'completion.json':dict(status='visual_stop_replan_resume_standoff_verified'),
            'inair-alignment.json':{},'prearm-alignment.json':{},'evidence-replay.json':{},'vision/accepted-request/request.json':{},
        }
        if kind=='controlled_stop':rows['stop-requested.json']={'reason':'operator stop'}
        for name,value in rows.items():
            path=out/name;path.parent.mkdir(parents=True,exist_ok=True);path.write_text(json.dumps(value))
        return out

    def test_both_current_outcomes_required_and_changed_runtime_invalidates(self):
        a=self.attempt('positive',1);write_qualification(a,'positive')
        self.assertFalse(qualification(self.config,self.digest)['qualified'])
        b=self.attempt('controlled_stop',2);write_qualification(b,'controlled_stop')
        self.assertTrue(qualification(self.config,self.digest)['qualified'])
        self.assertFalse(qualification(self.config,'c'*64)['qualified'])
        (b/'route.json').write_text('{}')
        self.assertFalse(qualification(self.config,self.digest)['qualified'])

    def test_no_qualification_for_ground_stop_or_failed_landing(self):
        out=self.attempt('controlled_stop',1)
        (out/'route.json').write_text(json.dumps({'samples':[{'position':{'vn':0,'ve':0}}]}))
        with self.assertRaisesRegex(ValueError,'horizontal motion'):write_qualification(out,'controlled_stop')
        receipt=json.loads((out/'runtime/receipt.json').read_text());receipt['final_armed']=True
        (out/'runtime/receipt.json').write_text(json.dumps(receipt))
        with self.assertRaisesRegex(ValueError,'Landing'):write_qualification(out,'controlled_stop')

    def test_missing_membership_cannot_be_replaced_by_status_claim(self):
        out=self.attempt('positive',1);value=write_qualification(out,'positive')
        del value['files']['vision/receipt.json'];value.pop('identity_sha256');value['identity_sha256']=object_sha256(value)
        (out/'qualification.json').write_text(json.dumps(value))
        result=qualification(self.config,self.digest)
        self.assertFalse(result['qualified']);self.assertIn('Incomplete',result['invalid'][0]['reason'])

    def test_path_escape_and_symlink_refused(self):
        for value in ('../flight','semantic-../x','semantic-'+'a'*31):
            with self.assertRaises(ValueError):run_directory(self.root,value)
        self.config.workbench_runs_root.mkdir()
        (self.config.workbench_runs_root/'escape').symlink_to(self.root,target_is_directory=True)
        with self.assertRaises(ValueError):model_directory(self.config,'escape')
        with self.assertRaises(ValueError):model_directory(self.config,'../escape')

    def test_managed_stop_targets_only_owned_run(self):
        job=SimpleNamespace(action='semantic-flight',scenario_id='semantic-'+'1'*32,job_id='one',error=None)
        self.assertTrue(cooperative_stop(self.config,job,lambda:True))
        out=run_directory(self.root,job.scenario_id)
        self.assertEqual(json.loads((out/'stop-requested.json').read_text())['job_id'],'one')
        job.scenario_id='../escape'
        with self.assertRaises(ValueError):cooperative_stop(self.config,job,lambda:True)

    def test_mission_refused_without_both_current_proofs(self):
        with patch('src.sandbox.semantic_commands.install_assets'), patch('src.sandbox.semantic_commands.current_identity',return_value={'identity_sha256':self.digest}):
            with self.assertRaisesRegex(ValueError,'qualification first'):
                build_semantic_command(self.config,None,{'model_id':'model','intent':'mission'})
        for parameters in ({'model_id':'model','intent':'anything'}, {'model_id':'model','intent':'mission','world':'new'}):
            with self.assertRaises(ValueError):build_semantic_command(self.config,None,parameters)
        self.config.profile='demo'
        with self.assertRaises(ValueError):build_semantic_command(self.config,None,{'model_id':'model','intent':'qualify-positive'})

    def test_port_conflict_is_not_misclassified_by_timeout_argument_in_traceback(self):
        from src.sandbox.failure_classification import classify_failure
        job=SimpleNamespace(state='failed',stop_requested=False,error='job exited with code 1',exit_code=1,
            diagnostics=['post_hover_timeout_s=180.', 'OSError: [Errno 48] Address already in use'])
        classify_failure(job)
        self.assertEqual(job.failure_code,'runtime_busy')

    def test_port_probe_distinguishes_active_listener_from_time_wait(self):
        import socket
        from src.flight.sitl_support import check_port_available
        with socket.socket() as server:
            server.setsockopt(socket.SOL_SOCKET,socket.SO_REUSEADDR,1)
            server.bind(('127.0.0.1',0));port=server.getsockname()[1];server.listen()
            with self.assertRaisesRegex(RuntimeError,'unavailable'):
                check_port_available(socket.SOCK_STREAM,port)
            client=socket.create_connection(('127.0.0.1',port))
            conn,_=server.accept();conn.close();client.recv(1);client.close()
        check_port_available(socket.SOCK_STREAM,port)
        with socket.socket(socket.AF_INET,socket.SOCK_DGRAM) as udp:
            udp.bind(('127.0.0.1',0))
            with self.assertRaisesRegex(RuntimeError,'unavailable'):
                check_port_available(socket.SOCK_DGRAM,udp.getsockname()[1])

    def test_asset_tamper_detected_without_accessing_original_paths(self):
        from src.sandbox.sitl_assets import ASSETS
        out=self.root/ASSETS;out.mkdir(parents=True);p=out/'world.sdf';p.write_text('fixed')
        from src.ml.artifacts import file_sha256
        with patch('src.sandbox.sitl_assets.MEMBERS',{'world.sdf':('/nonexistent/original/world',file_sha256(p))}):
            self.assertEqual(verify_assets(self.root),out)
            p.write_text('changed')
            with self.assertRaisesRegex(ValueError,'changed'):verify_assets(self.root)


if __name__=='__main__':unittest.main()

"""Final integration audit requires a managed flight AND an airborne UI stop."""
import argparse
from datetime import datetime,timezone
import json
import math
from pathlib import Path
from scripts.flight.check_px4_shadow import ROOT
from scripts.flight.diagnose_gps_health import read_samples
from scripts.flight.prepare_avoidance_route import collision_boxes
from scripts.flight.summarize_pillar_route import point_box_distance
from src.sandbox.live_replan_gate import BASE,require_repeat_gate,validate_record,POSITIVE
from src.sandbox.live_replan_stop import validate_run_id
from src.sandbox.job_models import SandboxJobStore
from src.sandbox.job_process import ownership_is_held,process_alive
from src.sandbox.workflow import inspect_workflow
from src.ml.artifacts import file_sha256,object_sha256
import sys
from src.vision.canonical.plan import write_record

def audit(success_job,stop_job):
    base=ROOT/BASE;out=base/'sandbox-integration-v1'
    if (out/'completion.json').exists():raise ValueError('Preserve existing integration audit')
    gate=require_repeat_gate(ROOT);store=SandboxJobStore(ROOT/'outputs/sandbox/operator/jobs')
    inputs=[];runs=[]
    for job_id,stopped in ((success_job,False),(stop_job,True)):
        job=store.read(job_id)
        if job.action!='live-replan-flight' or job.stop_requested!=stopped:raise ValueError('Unexpected managed action/stop identity')
        if process_alive(job.pid) or ownership_is_held(store.ownership_path(job_id),job.ownership_token):raise ValueError('Managed process still owned or alive')
        if not stopped and (job.state!='complete' or job.exit_code!=0):raise ValueError('Managed flight did not complete')
        if stopped and job.state!='failed':raise ValueError('Interrupted flight must not be marked complete')
        directory=base/validate_run_id(job.scenario_id)
        recipe_path=store.directory(job_id)/'workflow_recipe.json';receipt_path=store.directory(job_id)/'workflow_receipt.json'
        recipe=inspect_workflow(recipe_path,'recipe_identity_sha256');receipt=inspect_workflow(receipt_path,'receipt_identity_sha256')
        expected_command=[sys.executable,'-m','scripts.flight.fly_sandbox_replan','--fly','--run-id',job.scenario_id]
        if recipe['job_id']!=job_id or recipe['scenario_id']!=job.scenario_id or recipe['command_identity_sha256']!=object_sha256(expected_command):raise ValueError('Managed command identity conflict')
        if receipt['job_identity_sha256']!=job.to_record()['job_identity_sha256']:raise ValueError('Workflow references stale job state')
        process_path=store.process_result_path(job_id);process_result=json.loads(process_path.read_text())
        if process_result['exit_code']!=job.exit_code or process_result['ownership_token']!=job.ownership_token:raise ValueError('Managed process result mismatch')
        if receipt['state']!=('stopped' if stopped else 'complete') or receipt['recipe_identity_sha256']!=recipe['recipe_identity_sha256']:raise ValueError('Workflow state/identity conflict')
        for artifact in receipt['outputs']:
            if artifact['present'] and file_sha256(ROOT/artifact['path'])!=artifact['sha256']:raise ValueError('Managed output hash changed')
        protocol=validate_record(directory/'protocol.json');runtime=validate_record(directory/'runtime/receipt.json')
        if not runtime['landing_confirmed'] or runtime['final_armed'] is not False or not runtime['owned_processes_exited']:raise ValueError('Landing/disarm/cleanup incomplete')
        if stopped:
            marker=directory/'stop-requested.json';request=json.loads(marker.read_text())
            if request['job_id']!=job_id or 'Sandbox requested controlled stop and landing' not in runtime['error']:raise ValueError('No cooperative stop evidence')
            route_path=directory/'route.json';route=json.loads(route_path.read_text())
            if any(e['event']=='goal_reached_after_replan' for e in route['events']):raise ValueError('Stop test did not interrupt the route')
            rows=route['samples']
            if not rows or max(math.hypot(r['position']['vn'],r['position']['ve']) for r in rows)<=.1:raise ValueError('No actual cruise before stop')
            trigger_path=Path('/private/tmp')/f'uav-live-replan-stop-trigger-{job_id}.json';trigger=json.loads(trigger_path.read_text())
            if not route['replans'] or trigger['monotonic']<=route['replans'][0]['monotonic'] or trigger['phase']!='post_hover_trial' or math.hypot(trigger['value']['vn'],trigger['value']['ve'])<=.12:raise ValueError('Stop was not triggered after actual replanned cruise')
            with (directory/'runtime/telemetry.jsonl').open() as stream:
                if not any(json.loads(line)==trigger for line in stream):raise ValueError('Stop trigger is not original telemetry')
            if any(r['lidar_m']<=1.5 or r['cross_track_m']>.18 or math.hypot(r['position']['vn'],r['position']['ve'])>.35 for r in rows):raise ValueError('Stop trial flight guard violated')
            log=next((directory/'runtime/px4-work/log').rglob('*.ulg'))
            truth=[v for n,m,v in read_samples(log,{'vehicle_local_position_groundtruth'}) if m==0]
            envelope=base/'vehicle-envelope-002/envelope.json';radius=json.loads(envelope.read_text())['collision_sphere_radius_m']
            boxes=protocol['static_boxes']+collision_boxes(directory/'pillar.sdf')
            clearance=min(point_box_distance((v['y']+10,v['x']+10,-v['z']),b)-radius for v in truth for b in boxes)
            if clearance<=0:raise ValueError('Stop trial recorded collision envelope failed')
            metrics=dict(status='airborne_cooperative_stop_verified',whole_run_collision_sphere_clearance_m=clearance,landing_confirmed=True,final_armed=False,owned_processes_exited=True)
            from scripts.flight.replay_envelope_evidence import replay
            if not (directory/'evidence-replay.json').exists():replay(directory)
            validate_record(directory/'evidence-replay.json')
            inputs += [marker,trigger_path,route_path,log,envelope,directory/'pillar.sdf',directory/'runtime/telemetry.jsonl',directory/'evidence-replay.json']
        else:
            done=validate_record(directory/'completion.json')
            if done['status']!=POSITIVE:raise ValueError('Positive flight audit missing')
            for name in ('evidence-replay.json','map-registration.json','runtime/vision/completion.json'):validate_record(directory/name)
            metrics=done['metrics'];inputs.append(directory/'completion.json')
        inputs += [store.directory(job_id)/'job.json',recipe_path,receipt_path,process_path,directory/'protocol.json',directory/'runtime/receipt.json']
        for action in (('start','stop') if stopped else ('start',)):
            evidence=Path('/private/tmp')/f'uav-live-replan-{action}-{job_id}.json';record=json.loads(evidence.read_text())
            if record['job_id']!=job_id or record['scenario_id']!=job.scenario_id or record['errors'] or record['status']!=(202 if action=='start' else 200):raise ValueError('Desktop interaction evidence invalid')
            inputs.append(evidence)
        runs.append(dict(job_id=job_id,run_id=job.scenario_id,stopped=stopped,metrics=metrics))
    tests=out/'regressions-002.json';test_record=validate_record(tests)
    if test_record['status']!='focused_regressions_passed' or test_record['failures']:raise ValueError('Regression gate incomplete')
    for mode,disabled in [('locked',True),('ready',False)]:
        evidence=Path('/private/tmp')/f'uav-live-replan-{mode}-qa.json';record=json.loads(evidence.read_text())
        if record['errors'] or record['before']['disabled']!=disabled:raise ValueError('Desktop gate presentation invalid')
        inputs += [evidence,Path('/private/tmp')/f'uav-live-replan-{mode}-desktop.png']
    inputs += [tests,base/'startup-repeat-v2/completion.json',Path(__file__).resolve()]
    out.mkdir(exist_ok=True)
    report=out/'report.md'
    report.write_text('# 电脑端沙盒实时避障验收\n\n结论：固定开发场景的桌面沙盒闭环和中途受控停止均通过。不是任意地图、运动障碍或真机认证。\n\n'+
        '- 重复矩阵：6 次正向、1 次安全拒绝。复用低负载 v1 的 3 次正向及 1 次拒绝；新增启动门禁后补齐另 3 次，控制逻辑一致。原启动失败仍保留，不计通过。\n'+
        '- 桌面端：实际点击启动、确认、任务管理及停车降落；门禁未齐时 API 返回 400，未创建任务。遥测仅展示，不控制飞行。\n'+
        '- 已通过的沙盒完整飞行：`'+success_job+'`；中途停止：`'+stop_job+'`。停止任务按预期记录 stopped/failed，不冒充完整飞行。\n'+
        f'- 回归：{test_record["tests_run"]} 项，{test_record["passed"]} 项通过，{len(test_record["skipped"])} 项跳过；不是全仓测试声明。跳过原因见回执。\n'+
        '- 雷达仍采用 0.5 秒超时、1.5 米紧急门槛；规划采用 1.8+0.18+0.20 米误差预算；速度、姿态、漂移和重规划次数保护未放宽。\n'+
        '- 使用独立可行布局。原窄布局仍判不可行，未改称通过。新增静止柱的坐标未传给在线规划器；规划只使用实时雷达与已知静态地图。\n'+
        '- RGB640 为独立飞行负载配置，视觉模型仅旁路，无控制权；不用于 canonical 标注采集或模型精度认证。\n'+
        '- 之前雷达超时和启动超时的唯一根因尚未证实。静态对照、启动顺序修正和本轮重复通过是有限证据，不是永久可靠性保证。\n'+
        '- 只验收电脑端；按用户要求不扩展手机端。已知网站图标 404 不影响控制，未发现相关页面脚本错误。\n\n'+
        '下一优先方向：另行冻结更丰富的障碍几何、遮挡及真正运动目标测试矩阵；当前版本保持开发仿真用途。training_admitted=false，promotable=false。\n')
    inputs.append(report)
    write_record(out/'completion.json',dict(status='desktop_sandbox_live_replan_verified',created_at=datetime.now(timezone.utc).isoformat(),repeat_gate=gate,managed_runs=runs,desktop_only=True,physical_flight_certified=False,dynamic_object_tracking_verified=False,vision_control_authority='none',training_admitted=False,promotable=False,inputs={str(p.resolve()):file_sha256(p) for p in inputs}))
    print('DESKTOP_SANDBOX_LIVE_REPLAN_VERIFIED')

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--success-job',required=True);parser.add_argument('--stop-job',required=True);args=parser.parse_args();audit(args.success_job,args.stop_job)

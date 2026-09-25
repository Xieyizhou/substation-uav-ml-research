"""Offline report for the completed simulation-only heading profile."""
import bisect
import json
import math
from pathlib import Path
from scripts.flight.retest_sim_heading import OUT
from scripts.flight.diagnose_gps_health import read_samples
from src.ml.artifacts import file_sha256
from src.vision.canonical.plan import write_record


def main():
    runtime=OUT/'runtime'
    receipt=json.loads((runtime/'receipt.json').read_text())
    gate=json.loads((OUT/'prearm-alignment.json').read_text())
    rows=[json.loads(s) for s in (runtime/'telemetry.jsonl').read_text().splitlines()]
    times={e['event']:e['monotonic'] for e in receipt['events']}
    origin=receipt['origin']
    pos=[r['value'] for r in rows if r['stream']=='local' and times['takeoff_requested']<=r['monotonic']<=times['landing_confirmed']]
    ulog=next((runtime/'px4-work/log').rglob('*.ulg'))
    messages=list(read_samples(ulog,{'vehicle_local_position','vehicle_local_position_groundtruth'}))
    cutoff=gate['pairs'][-1]['timestamp_us']
    truth=[r for n,m,r in messages if n=='vehicle_local_position_groundtruth' and m==0]
    stamps=[r['timestamp'] for r in truth];heading=[];errors=[]
    f=gate['local_frame']
    for n,m,r in messages:
        if n!='vehicle_local_position' or m!=0 or r['timestamp']<cutoff:continue
        i=bisect.bisect_left(stamps,r['timestamp'])
        candidates=truth[max(0,i-1):i+1]
        t=min(candidates,key=lambda t:abs(t['timestamp']-r['timestamp']))
        if abs(t['timestamp']-r['timestamp'])>33334:continue
        heading.append(abs(math.degrees((r['heading']-t['heading']+math.pi)%(2*math.pi)-math.pi)))
        errors.append(math.hypot(r['y']+f['east_offset_m']-t['y']-10,r['x']+f['north_offset_m']-t['x']-10))
    metrics=dict(status=receipt['status'],prearm_max_heading_error_deg=gate['max_heading_error_deg'],
        post_gate_max_heading_error_deg=max(heading),post_gate_max_horizontal_registration_error_m=max(errors),
        hover_seconds=times['hover_complete']-times['hover_started'],
        maximum_height_m=max(origin['down']-p['down'] for p in pos),
        maximum_horizontal_drift_m=max(math.hypot(p['north']-origin['north'],p['east']-origin['east']) for p in pos),
        landing_confirmed=receipt['landing_confirmed'],final_armed=receipt['final_armed'],owned_processes_exited=receipt['owned_processes_exited'])
    text=f'''# 仿真航向修正与再次起飞

{json.dumps(metrics,ensure_ascii=False,indent=2)}

采用独立仿真参数：EKF2_DECL_TYPE=0、EKF2_MAG_DECL=-2.563032486°。磁偏角由此前模拟磁力计与仿真真值姿态推算，不是实体机标定值。模式0使用手动磁偏角的 PX4 源码逻辑已修正，未修改自动地理模型模式或健康阈值。前两次配置试验失败记录保留，均未解锁。

本次通过起飞前≤1°航向门禁，完成一次起飞、悬停和落地解除武装。post_gate 指核验窗口后至日志结束的只读真值比较，不是新的飞行验收门槛。41项针对性测试通过，不声明全仓通过。

证据支持仿真磁场与估计器磁偏角不一致，以及手动参数模式逻辑缺陷造成此前偏差。不把独立仿真配置用于实体机，不宣称已经修复全部 Gazebo 磁场坐标约定。

视觉始终只读；本轮没有执行横向自主航线、绕障或动态障碍规避。下一步仍需短航线与实际雷达刹停验证。training_admitted=false；promotable=false。
'''
    report=OUT/'report-zh.md'
    if report.exists():raise FileExistsError(report)
    report.write_text(text)
    paths=[ulog,runtime/'receipt.json',runtime/'telemetry.jsonl',runtime/'vision/completion.json',OUT/'prearm-alignment.json',OUT/'protocol.json',Path(__file__).resolve(),report]
    write_record(OUT/'completion.json',dict(metrics=metrics,inputs={str(p):file_sha256(p) for p in paths},training_admitted=False,promotable=False))
    print(text)


if __name__=='__main__':main()

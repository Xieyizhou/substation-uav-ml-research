"""Offline short-route validation using PX4 and simulator truth logs."""
import json
import math
from pathlib import Path
from scripts.flight.fly_short_route import OUT
from scripts.flight.diagnose_gps_health import read_samples
from src.ml.artifacts import file_sha256
from src.vision.canonical.plan import write_record


def main():
    route=json.loads((OUT/'route.json').read_text());flight=json.loads((OUT/'runtime/receipt.json').read_text())
    protocol=json.loads((OUT/'protocol.json').read_text());gate=json.loads((OUT/'inair-alignment.json').read_text())
    log=next((OUT/'runtime/px4-work/log').rglob('*.ulg'))
    truth=[r for n,m,r in read_samples(log,{'vehicle_local_position_groundtruth'}) if m==0 and r['timestamp']>gate['pairs'][-1]['timestamp_us']]
    cursor=0;hits=[]
    for east,north in protocol['map_targets_east_north_m']:
        match=None
        for i in range(cursor,len(truth)):
            r=truth[i];error=math.hypot(r['y']+10-east,r['x']+10-north)
            if error<=.15:
                match=dict(timestamp_us=r['timestamp'],map_east_m=r['y']+10,map_north_m=r['x']+10,error_m=error);cursor=i+1;break
        hits.append(match)
    samples=route['samples']
    metrics=dict(controller_reached=route['reached'],ordered_truth_proximity_checks=hits,
        truth_check_tolerance_m=.15,truth_check_role='independent postflight diagnostic, not a replacement for frozen .12m controller acceptance',
        maximum_measured_horizontal_speed_m_s=max(math.hypot(s['position']['vn'],s['position']['ve']) for s in samples),
        maximum_command_horizontal_speed_m_s=max(math.hypot(*s['command_ned'][:2]) for s in samples),
        minimum_lidar_distance_m=min(s['lidar_nearest_m'] for s in samples),
        landing_confirmed=flight['landing_confirmed'],final_armed=flight['final_armed'],owned_processes_exited=flight['owned_processes_exited'])
    complete=route['status']=='route_completed' and route['reached']==[0,1,2] and all(hits) and flight['status']=='sitl_hover_landed_disarmed' and flight['final_armed'] is False
    report=OUT/'report-zh.md'
    if report.exists():raise FileExistsError(report)
    report.write_text(f'''# PX4 短航线仿真验证

短航线完成并安全落地：{complete}。

{json.dumps(metrics,ensure_ascii=False,indent=2)}

同次地面和空中对齐门禁通过后，控制器在2米高度使用真实MAVSDK Offboard速度命令，按地图航点 (2,1.5)→(2,2)→(1.5,1.5) 去返，控制器分别满足0.12米及连续0.6秒条件。真值轨迹另做0.15米有序接近核对，不将其当成事先冻结的验收门槛。

0.3米/秒是指令上限；实际估计速度存在约0.017米/秒的短暂超调，不能声称实际速度始终≤0.3。雷达保护持续运行但本轮无近障碍，不证明保护已在运动中触发。未验证实际刹停、绕障或动态障碍避让。

视觉旁路同步运行，不参与控制。此次仅本机PX4/Gazebo，不连接实体机。下一步需要独立障碍刹停测试，随后才测试规划绕行。training_admitted=false；promotable=false。
''')
    paths=[OUT/'route.json',OUT/'protocol.json',OUT/'inair-alignment.json',OUT/'runtime/receipt.json',OUT/'runtime/vision/completion.json',log,report,Path(__file__).resolve()]
    write_record(OUT/'completion.json',dict(status='short_route_verified' if complete else 'requires_investigation',metrics=metrics,inputs={str(p):file_sha256(p) for p in paths},training_admitted=False,promotable=False))
    print(json.dumps(metrics,indent=2))


if __name__=='__main__':main()

"""Summarize an existing flight receipt; never launches simulation or flight."""
import json
import math
from pathlib import Path
import statistics
from src.ml.artifacts import file_sha256
from src.vision.canonical.plan import write_record


def main():
    root=Path(__file__).resolve().parents[2]
    out=root/'data/research/material-shadow-v1/px4-hover-flight-v1/attempt-004'
    receipt=json.loads((out/'receipt.json').read_text())
    rows=[json.loads(x) for x in (out/'telemetry.jsonl').read_text().splitlines()]
    events={x['event']:x['monotonic'] for x in receipt['events']}
    origin=receipt['origin']
    flight=[r for r in rows if events['takeoff_requested']<=r['monotonic']<=events['landing_confirmed']]
    hover=[r['value'] for r in rows if r['stream']=='local' and events['hover_started']<=r['monotonic']<=events['hover_complete']]
    positions=[r['value'] for r in flight if r['stream']=='local']
    predictions=[json.loads(x) for x in (out/'vision/detections.jsonl').read_text().splitlines()]
    in_flight=[p for p in predictions if events['takeoff_requested']<=p['result_monotonic']<=events['landing_confirmed']]
    delays=sorted(p['receive_to_logged_result_ms'] for p in in_flight)
    metrics=dict(hover_duration_s=events['hover_complete']-events['hover_started'],
        hover_height_min_m=min(origin['down']-p['down'] for p in hover),
        hover_height_max_m=max(origin['down']-p['down'] for p in hover),
        maximum_height_m=max(origin['down']-p['down'] for p in positions),
        maximum_horizontal_drift_m=max(math.hypot(p['north']-origin['north'],p['east']-origin['east']) for p in positions),
        observed_in_air=any(r['stream']=='in_air' and r['value'] is True for r in rows),
        final_armed=receipt['final_armed'],landing_confirmed=receipt['landing_confirmed'],
        flight_inferred_frames=len(in_flight),flight_receive_to_log_p50_ms=statistics.median(delays),
        flight_receive_to_log_p95_ms=delays[math.ceil(.95*len(delays))-1],
        true_capture_end_to_end_latency_verified=False)
    for path,digest in receipt['inputs'].items():
        if file_sha256(Path(path))!=digest:raise ValueError('Receipt input changed: '+path)
    inputs=[out/'receipt.json',out/'telemetry.jsonl',out/'vision/detections.jsonl',out/'vision/completion.json',Path(__file__).resolve()]
    text=f'''# PX4 仿真悬停试飞结果

状态：{receipt['status']}。仅本机 SITL + Gazebo；无实体无人机。

- GPS 未定义失败位修复后，不解锁复测的失败标志为 0，全局位置有效。
- attempt-003 在连接阶段超时，未解锁；attempt-004 完成一次飞行。
- 连续稳定悬停 {metrics['hover_duration_s']:.2f} 秒，高度 {metrics['hover_height_min_m']:.3f}–{metrics['hover_height_max_m']:.3f} 米（相对起飞前本地估计）。
- 起飞至落地期间最高 {metrics['maximum_height_m']:.3f} 米，最大水平漂移 {metrics['maximum_horizontal_drift_m']:.3f} 米。
- 已观测离地、确认落地和解除武装；本次进程已退出：{receipt['owned_processes_exited']}。
- 飞行窗口内视觉推理 {len(in_flight)} 帧，接收至结果日志前 P50 {metrics['flight_receive_to_log_p50_ms']:.1f} 毫秒，P95 {metrics['flight_receive_to_log_p95_ms']:.1f} 毫秒。

延迟从应用收到消息起算，不是经认证的相机采集到控制端到端延迟。视觉只记录检测，不控制飞行；未验证动态识别准确率、导航、避障、实体机安全性或复杂场景能力。一次仿真成功不等于模型取得飞行控制准入。训练及晋升标志保持 false。
'''
    report=out/'report-zh.md'
    if report.exists():raise FileExistsError(report)
    report.write_text(text)
    write_record(out/'analysis.json',dict(metrics=metrics,inputs={str(p):file_sha256(p) for p in inputs+[report]},training_admitted=False,promotable=False))
    print(text)


if __name__=='__main__':main()

/* Status is informational; the server rechecks the full flight gate on launch. */
(() => {
  let state = null, active = null;
  const start = document.querySelector('#live-replan-start');
  const stop = document.querySelector('#live-replan-stop');
  const badge = document.querySelector('#live-replan-badge');
  const summary = document.querySelector('#live-replan-summary');
  const telemetry = document.querySelector('#live-replan-telemetry');
  function render() {
    const ready = Boolean(state?.ready_for_launch_revalidation);
    start.disabled = !ready || Boolean(active) || profileState?.profile_id !== 'development';
    stop.disabled = active?.action !== 'live-replan-flight' || active?.state === 'stopping';
    badge.textContent = active?.action === 'live-replan-flight' ? active.state : ready ? '历史结果 · 需匹配冻结源码' : '历史记录未就绪';
    badge.className = 'badge warning';
    if (state) {
      const positive = state.units.filter(u => u.recorded_status === 'sensor_stop_replan_resume_goal_verified').length;
      const negative = state.units.filter(u => u.recorded_status === 'expected_safe_no_path_rejection_verified').length;
      summary.textContent = `历史已登记正向闭环 ${positive}/6；安全拒绝 ${negative}/1。${active?.action === 'live-replan-flight' ? '当前任务：' + active.job_id : ready ? '仅匹配原冻结源码时可复现；当前模型任务请使用上方入口。' : '等待全部重复测试通过。'}`;
      const run = state.latest_run;
      if (run) {
        const p = run.local_position;
        telemetry.textContent = `本入口任务 ${run.job_state} · 阶段 ${run.phase}` + (p ? ` · 局部 N/E ${p.north.toFixed(2)}/${p.east.toFixed(2)} m · 水平速度 ${run.speed_m_s.toFixed(2)} m/s · 遥测距今 ${run.telemetry_age_s.toFixed(1)} s` : ' · 等待遥测') + '。下方地图面板为另一流程，不表示本次轨迹。';
      }
    }
  }
  globalThis.renderLiveReplanOperator = job => { active = job; render(); };
  start.onclick = () => startManaged('live-replan-flight');
  stop.onclick = () => stopManaged(active?.job_id);
  async function refreshGate() {
    try { state = await api('/api/live-replan'); render(); }
    catch (error) { state = null; start.disabled = true; badge.textContent = '门禁读取失败'; summary.textContent = error.message; }
  }
  refreshGate(); setInterval(refreshGate, 5000);
})();

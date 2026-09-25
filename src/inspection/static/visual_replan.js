/* The backend verifies source and evidence identities again before launch. */
(() => {
  let state = null, active = null;
  const start = document.querySelector('#visual-replan-start');
  const stop = document.querySelector('#visual-replan-stop');
  const badge = document.querySelector('#visual-replan-badge');
  const summary = document.querySelector('#visual-replan-summary');
  const telemetry = document.querySelector('#visual-replan-telemetry');
  function render() {
    const ready = Boolean(state?.ready_for_launch_revalidation);
    const running = active?.action === 'visual-replan-flight';
    start.disabled = !ready || Boolean(active) || profileState?.profile_id !== 'development';
    stop.disabled = !running || active?.state === 'stopping';
    badge.textContent = running ? active.state : ready ? '可运行 · 启动前核验' : '等待验证通过';
    badge.className = `badge ${ready ? 'pass' : 'warning'}`;
    if (state) {
      summary.textContent = `视觉改路线验证${state.positive_verified ? '已通过' : '待完成'}；中途停车降落验证${state.controlled_stop_verified ? '已通过' : '待完成'}。`;
      const run = state.latest_run;
      if (run) {
        const p = run.local_position;
        telemetry.textContent = `本入口任务 ${run.job_state} · 阶段 ${run.phase}` + (p ? ` · 局部 N/E ${p.north.toFixed(2)}/${p.east.toFixed(2)} m · 水平速度 ${run.speed_m_s.toFixed(2)} m/s` : ' · 等待遥测');
      }
    }
  }
  globalThis.renderVisualReplanOperator = job => { active = job; render(); };
  start.onclick = async () => {
    const accepted = await requestConfirmation({title:'开始视觉改路线仿真？',
      detail:'识别设备，停车后生成观察路线，继续飞行并降落。',
      facts:{场景:'现有固定场景',任务:'设备观察',运行方式:'本机 PX4 / Gazebo 仿真'},confirmLabel:'开始仿真'});
    if (!accepted) return;
    try { await post('/api/operator/start',{action:'visual-replan-flight'}); await refreshOperator(); }
    catch (error) { actionError(error.message); }
  };
  stop.onclick = () => stopManaged(active?.job_id);
  async function refresh() {
    try { state = await api('/api/visual-replan'); render(); }
    catch (error) { state = null; start.disabled = true; badge.textContent = '验证记录读取失败'; summary.textContent = error.message; }
  }
  refresh(); setInterval(refresh, 5000);
})();

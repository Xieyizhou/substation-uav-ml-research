/* Historical evidence is independent of current runtime qualification. */
(() => {
  let state = null, active = null;
  const button = document.querySelector('#evidence-verify');
  const badge = document.querySelector('#evidence-badge');
  const detail = document.querySelector('#evidence-summary');
  function render() {
    const running = active?.action === 'evidence-verify';
    button.disabled = !state?.available || !operatorToken || Boolean(active);
    badge.textContent = running ? '正在核验完整证据包' : !state?.available ? '未安装证据包' : state?.last_verification ? '已有历史核验结果' : '等待核验';
    badge.className = `badge ${!running && state?.last_verification && !state.changed_since_verification ? 'pass' : 'warning'}`;
    if (!state) return;
    const report = state.last_verification;
    detail.textContent = report ? `上次核验：${new Date(report.verified_at).toLocaleString()}。${report.records} 份记录；6 次 LiDAR 正向飞行、1 次无路拒绝、1 次视觉改线和1次中途停止。${state.changed_since_verification ? '证据包此后有变化，请重新核验。' : ''}` : state.reason;
  }
  globalThis.renderEvidenceOperator = job => { active = job; render(); };
  button.onclick = async () => {
    button.disabled = true;
    try { await post('/api/operator/start', {action: 'evidence-verify'}); await refreshOperator(); }
    catch (error) { actionError(error.message); render(); }
  };
  async function refresh() {
    try { state = await api('/api/evidence'); render(); }
    catch (error) { button.disabled = true; badge.textContent = '读取失败'; detail.textContent = error.message; }
  }
  refresh(); setInterval(refresh, 5000);
})();

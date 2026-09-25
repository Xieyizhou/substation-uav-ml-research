/* Launch always rehashes the current runtime and selected model. */
(() => {
  const q = id => document.getElementById('semantic-'+id);
  let active = null, state = null, checked = null, sequence = 0, busy = false;
  function render() {
    const running = active?.action === 'semantic-flight';
    const available = checked?.available && q('model').value && !active && !busy && operatorToken && profileState?.profile_id === 'development';
    q('positive').disabled = !available;
    q('stop-trial').disabled = !available;
    q('run').disabled = !available || !checked?.qualified;
    q('model').disabled = Boolean(active) || busy;
    q('stop').disabled = !running || active.state === 'stopping';
    q('badge').textContent = running ? active.state : busy ? '正在核验当前输入' : checked?.qualified ? '当前模型与运行代码已验收' : '需要当前运行验收';
    q('badge').className = `badge ${checked?.qualified ? 'pass' : 'warning'}`;
    if (checked) q('status').textContent = checked.available ? `完整任务：${checked.evidence.positive ? '通过' : '待验收'}；中途停止：${checked.evidence.controlled_stop ? '通过' : '待验收'}。运行身份 ${checked.runtime_identity_sha256.slice(0,12)}。启动前将重新核验。` : checked.reason;
    const run = state?.latest_run;
    if (run) q('telemetry').textContent = `${run.model_id || ''} · ${run.intent || ''} · ${run.qualification_kind === 'controlled_stop' ? '已按要求停止并降落' : run.state} · ${run.phase}` + (run.speed_m_s == null ? '' : ` · 水平速度 ${run.speed_m_s.toFixed(2)} m/s`) + (run.qualification_kind ? ` · ${run.qualification_kind} 验收通过` : '') + (run.completion ? ' · 观察任务已完成' : '');
  }
  async function check() {
    const ticket = ++sequence, id = q('model').value;
    checked = null; busy = Boolean(id); render();
    if (!id) { q('status').textContent = '请选择已验证模型。'; return; }
    try { const result = await api('/api/semantic-model/'+encodeURIComponent(id)); if (ticket === sequence) checked = result; }
    catch (error) { if (ticket === sequence) checked = {available:false,reason:error.message}; }
    finally { if (ticket === sequence) { busy = false; render(); } }
  }
  globalThis.renderSemanticOperator = job => {
    const ended = active?.action === 'semantic-flight' && !job;
    active = job; render(); if (ended) check();
  };
  q('model').onchange = check;
  async function launch(intent) {
    const model_id = q('model').value;
    const accepted = await requestConfirmation({title: intent === 'mission' ? '执行已验收观察任务？' : '开始当前模型的仿真验收？',
      detail: intent === 'qualify-stop' ? '开始水平飞行后，点击“停车并降落”完成停止验收。' : '识别设备、停车改线、到达观察点并降落。',
      facts:{模型:model_id,场景:'现有固定场景',范围:'仅本机 PX4 / Gazebo'},confirmLabel:'开始仿真'});
    if (!accepted) return;
    busy = true; render();
    try { await post('/api/operator/start',{action:'semantic-flight',parameters:{model_id,intent}}); await refreshOperator(); }
    catch (error) { actionError(error.message); }
    finally { busy = false; render(); }
  }
  q('positive').onclick = () => launch('qualify-positive');
  q('stop-trial').onclick = () => launch('qualify-stop');
  q('run').onclick = () => launch('mission');
  q('stop').onclick = () => stopManaged(active?.job_id);
  async function refresh() {
    try {
      state = await api('/api/semantic-flight');
      const selected = q('model').value;
      const signature = state.models.map(m=>m.experiment_id).join('|');
      if (q('model').dataset.signature !== signature) {
        q('model').replaceChildren(new Option('选择模型',''), ...state.models.map(m=>new Option(m.experiment_id,m.experiment_id)));
        q('model').dataset.signature = signature;
        if (state.models.some(m=>m.experiment_id===selected)) q('model').value = selected;
        else { checked = null; sequence++; }
      }
      render();
    } catch (error) { q('telemetry').textContent = error.message; }
  }
  refresh(); setInterval(refresh,3000);
})();

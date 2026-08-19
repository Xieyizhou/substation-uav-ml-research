(function(root) {
  const q = id => document.getElementById(id);
  const {colors, node} = root.MapCanvas;

  function selection() {
    return JSON.parse(root.sessionStorage.getItem('sandboxMapFlight') || 'null');
  }

  function renderSelection() {
    const value = selection();
    q('map-flight-start').disabled = !value;
    q('map-record-start').disabled = !value;
    q('map-flight-facts').innerHTML = value ?
      `<span>Map<b>${esc(value.map_name)}</b></span>` +
      `<span>Mission<b>${esc(value.mission_id)}</b></span>` +
      `<p>Revision ${esc(value.revision_id.slice(0,16))}…</p>` :
      '<p>Select a validated map revision in Map Studio.</p>';
  }

  function flightSvg(value) {
    const svg = q('map-flight-canvas');
    const map = value.map;
    const route = value.route;
    if (!map || !route) {
      svg.innerHTML = '';
      return;
    }
    const scaleX = 600 / map.width_m;
    const scaleY = 420 / map.height_m;
    const point = (east, north) => [east * scaleX, 420 - north * scaleY];
    const gridPath = rows => rows.map((cell, index) => {
      const location = point(cell[0] + 0.5, cell[1] + 0.5);
      return `${index ? 'L' : 'M'} ${location[0]} ${location[1]}`;
    }).join(' ');
    svg.innerHTML = '';
    svg.append(node('rect', {
      x: 0, y: 0, width: 600, height: 420, fill: '#edf0eb',
    }));
    map.objects.forEach(item => {
      const location = point(item.east_m, item.north_m);
      svg.append(node('rect', {
        x: location[0] - item.width_m * scaleX / 2,
        y: location[1] - item.depth_m * scaleY / 2,
        width: item.width_m * scaleX,
        height: item.depth_m * scaleY,
        fill: colors[item.asset_id],
        opacity: 0.9,
      }));
    });
    svg.append(node('path', {
      d: gridPath(route.grid_path),
      fill: 'none',
      stroke: '#75837e',
      'stroke-width': 3,
      'stroke-dasharray': '7 5',
    }));
    const actual = (value.trajectory || []).map((item, index) => {
      const location = point(item.east_m, item.north_m);
      return `${index ? 'L' : 'M'} ${location[0]} ${location[1]}`;
    }).join(' ');
    if (actual) {
      svg.append(node('path', {
        d: actual, fill: 'none', stroke: '#0b8062', 'stroke-width': 4,
      }));
    }
    const live = value.live;
    if (live?.position?.east_m == null) return;
    const location = point(live.position.east_m, live.position.north_m);
    const drone = node('g', {
      transform: `translate(${location[0]} ${location[1]}) rotate(${live.yaw_deg || 0})`,
    });
    drone.append(node('circle', {
      r: 8,
      fill: live.stale ? '#b06c24' : '#08745c',
      stroke: '#fff',
      'stroke-width': 2,
    }));
    drone.append(node('path', {
      d: 'M 0 -14 L -4 -5 L 4 -5 Z', fill: '#08745c',
    }));
    svg.append(drone);
  }

  function renderRecordingAudit(value) {
    const audit = value.recording_audit;
    q('map-record-register').disabled = true;
    if (audit?.valid) {
      q('map-recording-audit').innerHTML =
        '<p><span class="badge pass">Audited</span>' +
        `<strong>${audit.frame_count.toLocaleString()} manifest frames · ` +
        `${audit.labelled_frame_count.toLocaleString()} labelled · ` +
        `${audit.no_target_frame_count.toLocaleString()} no-target</strong>` +
        (audit.orphan_payload_count ?
          `<br><span>${audit.orphan_payload_count.toLocaleString()} ` +
          'unreferenced PNG payloads ignored</span>' : '') +
        'Ready to register as a development dataset.</p>';
      q('map-record-register').disabled = value.active;
      q('map-record-register').dataset.run = value.latest.run_root.split('/').at(-1);
    } else if (audit) {
      q('map-recording-audit').innerHTML =
        `<p><span class="badge failure">Recording invalid</span>` +
        `${esc(audit.error || 'Recording audit failed.')}</p>`;
    } else {
      q('map-recording-audit').innerHTML =
        '<p>PNG recording is optional. A completed, audited recording can be ' +
        'registered as a development dataset.</p>';
    }
  }

  function renderFlightFacts(value) {
    const live = value.live;
    const latest = value.latest;
    const health = value.active ?
      (live?.stale ? 'Telemetry stale' : 'Live') :
      (latest.status || latest.state);
    const healthClass = health === 'Live' || health === 'complete' ?
      'pass' : (health === 'Telemetry stale' ? 'warning' : 'failure');
    q('map-flight-health').className = `badge ${healthClass}`;
    q('map-flight-health').textContent = health;
    const speed = live ? Math.hypot(
      live.velocity.north_m_s || 0, live.velocity.east_m_s || 0,
    ) : null;
    const altitude = live?.position?.down_m == null ?
      '—' : `${Math.max(0, -live.position.down_m).toFixed(2)} m`;
    const elapsed = live?.elapsed_s == null ?
      '—' : `${live.elapsed_s.toFixed(1)} s`;
    q('map-flight-facts').innerHTML =
      `<span>Map<b>${esc(latest.map_id)}</b></span>` +
      `<span>Phase<b>${esc(live?.phase || latest.status)}</b></span>` +
      `<span>Speed<b>${speed == null ? '—' : `${speed.toFixed(2)} m/s`}</b></span>` +
      `<span>Altitude<b>${altitude}</b></span>` +
      `<span>Elapsed<b>${elapsed}</b></span>` +
      `<span>Telemetry<b>${live?.stale ? 'stale' : 'healthy'}</b></span>`;
  }

  async function refresh() {
    renderSelection();
    try {
      const value = await api('/api/map-runs');
      const hasSelection = Boolean(selection());
      q('map-record-start').disabled = value.active || !hasSelection;
      if (!value.latest) {
        q('map-flight-health').className = 'badge warning';
        q('map-flight-health').textContent = 'No map run';
        return;
      }
      flightSvg(value);
      renderFlightFacts(value);
      renderRecordingAudit(value);
      q('map-flight-stop').disabled = !value.active;
      q('map-flight-start').disabled = value.active || !hasSelection;
    } catch (error) {
      q('map-flight-health').textContent = error.message;
    }
  }

  function startSelected(action) {
    const value = selection();
    if (!value) return;
    startManaged(action, null, {
      map_id: value.map_id,
      revision_id: value.revision_id,
      mission_id: value.mission_id,
      display_mode: q('map-flight-display').value,
    });
  }

  async function registerRecording() {
    const runId = q('map-record-register').dataset.run;
    if (!runId) return;
    const accepted = await requestConfirmation({
      title: 'Register audited recording?',
      detail: 'The recording will become an immutable development dataset in Dataset Manager.',
      facts: {Run: runId, Role: 'development'},
      confirmLabel: 'Register dataset',
    });
    if (!accepted) return;
    try {
      await post('/api/maps/register', {run_id: runId});
      actionError('Custom-map dataset registered.');
      if (typeof root.refresh === 'function') await root.refresh();
      activateRoute('model/datasets');
    } catch (error) {
      actionError(error.message);
    }
  }

  async function poll() {
    await refresh();
    setTimeout(poll, q('map-flight-stop').disabled ? 3000 : 500);
  }

  q('map-flight-start').onclick = () => startSelected('map-flight-smoke');
  q('map-record-start').onclick = () => startSelected('map-record');
  q('map-flight-stop').onclick = () => stopManaged(q('operator-stop').dataset.job);
  q('map-record-register').onclick = registerRecording;
  root.MapFlight = {refresh, renderSelection};
  refresh();
  setTimeout(poll, 1000);
})(typeof globalThis === 'undefined' ? this : globalThis);

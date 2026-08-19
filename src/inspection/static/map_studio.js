(function(root) {
  const state = {
      catalog: null,
      detail: null,
      map: null,
      selected: null,
      source: null,
      undo: [],
      redo: [],
      revision: null,
      drag: null
    };
  const q = id => document.getElementById(id),
    snap = value => Math.round(Number(value) * 2) / 2,
    clone = value => JSON.parse(JSON.stringify(value));
  const {colors} = root.MapCanvas;

  function selectedObject() {
    return state.map?.objects.find(item => item.object_id === state.selected) || null
  }

  function selectedMission() {
    return state.map?.missions.find(item => item.mission_id === q('map-mission').value) || state.map?.missions[0] || null
  }

  function remember() {
    state.undo.push(clone(state.map));
    if (state.undo.length > 40) state.undo.shift();
    state.redo = []
  }

  function changed() {
    state.revision = null;
    q('map-export').hidden = true;
    q('map-send-flight').disabled = true;
    draw();
    scheduleSave()
  }
  let saveTimer = null;

  function scheduleSave() {
    clearTimeout(saveTimer);
    saveTimer = setTimeout(() => saveDraft(true), 900)
  }

  function listMaps() {
    const rows = [...(state.catalog?.drafts || []).map(map => ({
      map,
      kind: 'Draft'
    })), ...(state.catalog?.templates || []).map(map => ({
      map,
      kind: 'Template'
    }))];
    q('map-list').innerHTML = rows.map(row => `<button data-map="${esc(row.map.map_id)}" class="${state.map?.map_id===row.map.map_id?'active':''}"><span><b>${esc(row.map.display_name)}</b><small>${esc(row.map.width_m)} × ${esc(row.map.height_m)} m</small></span><em>${row.kind}</em></button>`).join('');
    q('map-list').querySelectorAll('button').forEach(button => button.onclick = () => loadMap(button.dataset.map))
  }

  function assets() {
    q('map-assets').innerHTML = (state.catalog?.assets || []).map((item, index) => `<button data-asset="${esc(item.asset_id)}"><span class="asset-symbol" style="background:${colors[item.asset_id]}">${index+1}</span>${esc(item.display_name)}</button>`).join('');
    q('map-assets').querySelectorAll('button').forEach(button => button.onclick = () => addAsset(button.dataset.asset))
  }

  function mapInputs() {
    q('map-name').value = state.map.display_name;
    q('map-width').value = state.map.width_m;
    q('map-height').value = state.map.height_m;
    q('map-start-east').value = state.map.start_east_m;
    q('map-start-north').value = state.map.start_north_m;
    const selected = q('map-mission').value;
    q('map-mission').innerHTML = state.map.missions.map(item => `<option value="${esc(item.mission_id)}">${esc(item.mission_id.replaceAll('_',' '))}</option>`).join('');
    if (state.map.missions.some(item => item.mission_id === selected)) q('map-mission').value = selected;
    missionInputs();
    objectInputs()
  }

  function objectInputs() {
    const item = selectedObject(),
      panel = q('object-properties');
    panel.querySelector('.empty-selection').hidden = !!item;
    panel.querySelector('.selected-properties').hidden = !item;
    if (!item) return;
    const fields = {
      id: item.object_id,
      asset: item.asset_id,
      east: item.east_m,
      north: item.north_m,
      width: item.width_m,
      depth: item.depth_m,
      height: item.height_m,
      yaw: item.yaw_deg
    };
    Object.entries(fields).forEach(([name, value]) => q(`object-${name}`).value = value)
  }

  function missionInputs() {
    const item = selectedMission();
    if (!item) return;
    q('mission-type').value = item.mission_type;
    q('mission-east').value = item.goal_east_m ?? '';
    q('mission-north').value = item.goal_north_m ?? '';
    q('mission-altitude').value = item.altitude_m;
    q('mission-speed').value = item.speed_m_s;
    const targets = state.map.objects.filter(row => row.label_role === 'target');
    q('mission-target').innerHTML = targets.map(row => `<option value="${esc(row.object_id)}">${esc(row.object_id)}</option>`).join('');
    q('mission-target').value = item.target_object_id || targets[0]?.object_id || '';
    const inspection = item.mission_type === 'equipment_inspection';
    q('mission-point-fields').hidden = inspection;
    q('mission-target-field').hidden = !inspection
  }

  function draw() {
    if (!state.map) return;
    const svg = q('map-canvas'),
      map = state.map,
      route = state.detail?.preview?.routes?.find(item => item.mission_id === selectedMission()?.mission_id);
    root.MapCanvas.render({
      svg, map, route, selected: state.selected,
      overlay: q('map-overlay').value,
      onPointerDown: beginDrag,
    });
    mapInputs();
    listMaps()
  }

  function point(event) {
    return root.MapCanvas.eventPoint(q('map-canvas'), state.map, event, snap)
  }

  function beginDrag(event, id) {
    event.preventDefault();
    state.selected = id;
    remember();
    state.drag = id;
    q('map-canvas').setPointerCapture(event.pointerId);
    objectInputs();
    draw()
  }
  q('map-canvas').addEventListener('pointermove', event => {
    if (!state.drag) return;
    const item = selectedObject(),
      [east, north] = point(event);
    item.east_m = Math.max(0, Math.min(state.map.width_m, east));
    item.north_m = Math.max(0, Math.min(state.map.height_m, north));
    draw()
  });
  q('map-canvas').addEventListener('pointerup', () => {
    if (state.drag) {
      state.drag = null;
      changed()
    }
  });
  q('map-canvas').addEventListener('pointercancel', () => {
    state.drag = null
  });

  function addAsset(assetId) {
    if (!state.map) return;
    remember();
    const asset = state.catalog.assets.find(item => item.asset_id === assetId),
      count = state.map.objects.filter(item => item.asset_id === assetId).length + 1,
      id = `${assetId}_${count}`;
    state.map.objects.push({
      object_id: id,
      asset_id: assetId,
      east_m: snap(state.map.width_m / 2),
      north_m: snap(state.map.height_m / 2),
      width_m: asset.default_size_m[0],
      depth_m: asset.default_size_m[1],
      height_m: asset.default_size_m[2],
      yaw_deg: 0,
      label_role: asset.label_role
    });
    state.selected = id;
    changed()
  }

  async function loadMap(id) {
    try {
      state.detail = await api(`/api/maps/${encodeURIComponent(id)}`);
      state.map = clone(state.detail.map);
      state.source = state.detail.source;
      state.selected = null;
      state.undo = [];
      state.redo = [];
      state.revision = (state.catalog?.revisions || []).find(item =>
        item.map_id === state.map.map_id &&
        item.map_identity_sha256 === state.map.map_identity_sha256
      ) || null;
      if (state.revision) {
        const revisionId = state.revision.revision_identity_sha256;
        q('map-export').hidden = false;
        q('map-export').href =
          `/api/maps/bundle/${encodeURIComponent(state.map.map_id)}/${revisionId}`;
        selectForFlight();
      } else {
        q('map-export').hidden = true;
      }
      renderValidation();
      draw()
    } catch (error) {
      actionError(error.message)
    }
  }
  async function saveDraft(quiet = false) {
    if (!state.map) return;
    try {
      if (state.source === 'template') {
        state.map.map_id = `custom_${Date.now().toString().slice(-8)}`;
        state.map.display_name = `${state.map.display_name} copy`;
        state.source = 'draft'
      }
      state.detail = await post('/api/maps/draft', {
        map: state.map
      });
      state.map = clone(state.detail.map);
      state.catalog = await api('/api/maps');
      renderValidation();
      draw();
      if (!quiet) actionError('Draft saved and routes revalidated.')
    } catch (error) {
      renderInvalid(error.message)
    }
  }

  function renderInvalid(message) {
    q('map-validation').innerHTML = `<span class="badge failure">Invalid</span><span>${esc(message)}</span>`;
    q('map-send-flight').disabled = true
  }

  function renderValidation() {
    const report = state.detail?.validation;
    if (!report) return;
    const errors = report.issues.filter(item => item.severity === 'error');
    q('map-validation').innerHTML = `<span class="badge ${report.valid?'pass':'failure'}">${report.valid?'Valid':'Needs work'}</span><span>${report.valid?`${report.checks_passed}/${report.checks_total} checks passed`:esc(errors[0]?.message||'Validation failed')}</span>`;
    const route = state.detail.preview.routes.find(item => item.mission_id === selectedMission()?.mission_id);
    q('map-route-summary').textContent = route ? `${route.simplified_path.length} route points · about ${Math.ceil(route.estimated_duration_s)} s` : 'No valid route preview';
    q('map-revision').disabled = !report.valid;
    q('map-send-flight').disabled = !state.revision
  }
  async function createRevision() {
    await saveDraft(true);
    if (!state.detail?.validation?.valid) return;
    try {
      const result = await post('/api/maps/revision', {
        map_id: state.map.map_id
      });
      state.revision = result.revision;
      const id = result.revision.revision_identity_sha256;
      q('map-export').hidden = false;
      q('map-export').href = `/api/maps/bundle/${encodeURIComponent(state.map.map_id)}/${id}`;
      q('map-send-flight').disabled = false;
      selectForFlight();
      actionError('Immutable map revision created.')
    } catch (error) {
      renderInvalid(error.message)
    }
  }

  function selectForFlight() {
    if (!state.revision) return;
    const mission = selectedMission();
    root.sessionStorage.setItem('sandboxMapFlight', JSON.stringify({
      map_id: state.map.map_id,
      revision_id: state.revision.revision_identity_sha256,
      mission_id: mission.mission_id,
      map_name: state.map.display_name
    }));
    if (root.MapFlight) root.MapFlight.renderSelection()
  }
  async function importBundle(event) {
    const file = event.target.files[0];
    if (!file) return;
    const bytes = new Uint8Array(await file.arrayBuffer());
    let binary = '';
    bytes.forEach(value => binary += String.fromCharCode(value));
    try {
      const result = await post('/api/maps/import', {
        bundle_base64: btoa(binary)
      });
      state.catalog = await api('/api/maps');
      await loadMap(result.map.map_id)
    } catch (error) {
      actionError(error.message)
    }
    event.target.value = ''
  }
  async function refresh() {
    try {
      state.catalog = await api('/api/maps');
      assets();
      const selected = state.map?.map_id || state.catalog.drafts[0]?.map_id || state.catalog.templates[0]?.map_id;
      if (selected) await loadMap(selected);
      if (root.MapFlight) await root.MapFlight.refresh()
    } catch (error) {
      q('map-list').textContent = error.message
    }
  }

  root.MapStudioActions.bind({
    q, state, clone, snap, selectedObject, selectedMission, remember, changed,
    draw, scheduleSave, saveDraft, createRevision, refresh, importBundle,
    selectForFlight, missionInputs
  });
  root.MapStudio = {
    refresh
  };
  refresh();
})(typeof globalThis === 'undefined' ? this : globalThis);

(function(root) {
  function bind(ctx) {
    const {
      q, state, clone, snap, selectedObject, selectedMission, remember,
      changed, draw, scheduleSave, saveDraft, createRevision, refresh,
      importBundle, selectForFlight, missionInputs
    } = ctx;

    function blankMap() {
      const stamp = Date.now().toString().slice(-8);
      return {
        map_id: `custom_${stamp}`, display_name: 'Untitled substation',
        width_m: 24, height_m: 24, start_east_m: 1.5, start_north_m: 1.5,
        start_yaw_deg: 0, objects: [], resolution_m: 1, editor_snap_m: .5,
        weather: 'clear', light_level: 1, camera_noise_stddev: 0,
        parent_revision_identity: null, sandbox_map_schema_version: 1,
        missions: [{
          mission_id: 'default_round_trip', mission_type: 'round_trip',
          goal_east_m: 20.5, goal_north_m: 20.5, target_object_id: null,
          altitude_m: 1.5, speed_m_s: 1, horizontal_inflation_cells: 1
        }]
      }
    }

    function updateObject() {
      const item = selectedObject();
      if (!item) return;
      remember();
      ['east', 'north', 'width', 'depth', 'height'].forEach(name => {
        item[`${name}_m`] = Number(q(`object-${name}`).value)
      });
      item.yaw_deg = Number(q('object-yaw').value);
      changed()
    }

    function updateMap() {
      remember();
      state.map.display_name = q('map-name').value;
      ['width', 'height'].forEach(name => {
        state.map[`${name}_m`] = Number(q(`map-${name}`).value)
      });
      state.map.start_east_m = Number(q('map-start-east').value);
      state.map.start_north_m = Number(q('map-start-north').value);
      changed()
    }

    function updateMission() {
      const item = selectedMission();
      if (!item) return;
      remember();
      item.mission_type = q('mission-type').value;
      item.altitude_m = Number(q('mission-altitude').value);
      item.speed_m_s = Number(q('mission-speed').value);
      if (item.mission_type === 'equipment_inspection') {
        item.target_object_id = q('mission-target').value;
        item.goal_east_m = null;
        item.goal_north_m = null
      } else {
        item.target_object_id = null;
        item.goal_east_m = Number(q('mission-east').value);
        item.goal_north_m = Number(q('mission-north').value)
      }
      changed()
    }

    ['map-name', 'map-width', 'map-height', 'map-start-east', 'map-start-north']
      .forEach(id => q(id).addEventListener('change', updateMap));
    ['object-east', 'object-north', 'object-width', 'object-depth', 'object-height', 'object-yaw']
      .forEach(id => q(id).addEventListener('change', updateObject));
    ['mission-type', 'mission-east', 'mission-north', 'mission-target', 'mission-altitude', 'mission-speed']
      .forEach(id => q(id).addEventListener('change', updateMission));
    q('map-mission').onchange = () => { missionInputs(); draw() };
    q('map-overlay').onchange = draw;
    q('map-new').onclick = () => {
      state.map = blankMap(); state.detail = null; state.source = 'draft';
      state.selected = null; state.revision = null; draw(); scheduleSave()
    };
    q('map-save').onclick = () => saveDraft();
    q('map-revision').onclick = createRevision;
    q('map-delete').onclick = async () => {
      if (state.source !== 'draft') return actionError('Templates are read-only.');
      try {
        await post('/api/maps/draft/delete', {map_id: state.map.map_id});
        state.map = null; await refresh()
      } catch (error) { actionError(error.message) }
    };
    q('map-undo').onclick = () => {
      if (!state.undo.length) return;
      state.redo.push(clone(state.map)); state.map = state.undo.pop(); changed()
    };
    q('map-redo').onclick = () => {
      if (!state.redo.length) return;
      state.undo.push(clone(state.map)); state.map = state.redo.pop(); changed()
    };
    q('object-remove').onclick = () => {
      if (!state.selected) return;
      remember();
      state.map.objects = state.map.objects.filter(item => item.object_id !== state.selected);
      state.selected = null; changed()
    };
    q('object-duplicate').onclick = () => {
      const item = selectedObject();
      if (!item) return;
      remember();
      const copy = clone(item);
      copy.object_id = `${item.asset_id}_${Date.now().toString().slice(-4)}`;
      copy.east_m = snap(item.east_m + 1); copy.north_m = snap(item.north_m + 1);
      state.map.objects.push(copy); state.selected = copy.object_id; changed()
    };
    q('map-import').onchange = importBundle;
    q('map-send-flight').onclick = () => { selectForFlight(); activateRoute('fly/run') }
  }

  root.MapStudioActions = {bind}
})(typeof globalThis === 'undefined' ? this : globalThis);

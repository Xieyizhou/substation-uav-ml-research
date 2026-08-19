(function(root) {
  const ns = 'http://www.w3.org/2000/svg';
  const colors = {
    transformer: '#234d55',
    switchgear: '#217b85',
    capacitor_bank: '#4e8c80',
    reactor: '#59636b',
    cabinet: '#16829a',
    pole: '#4d5552',
    control_building: '#777b78',
    generic_obstacle: '#88533d',
  };

  function node(name, attrs = {}) {
    const value = document.createElementNS(ns, name);
    Object.entries(attrs).forEach(([key, item]) => value.setAttribute(key, item));
    return value;
  }

  function linePath(cells, map) {
    return (cells || []).map((cell, index) =>
      `${index ? 'L' : 'M'} ${cell[0] + .5} ${map.height_m - cell[1] - .5}`
    ).join(' ');
  }

  function eventPoint(svg, map, event, snap) {
    const point = svg.createSVGPoint();
    point.x = event.clientX;
    point.y = event.clientY;
    const result = point.matrixTransform(svg.getScreenCTM().inverse());
    return [snap(result.x), snap(map.height_m - result.y)];
  }

  function render(options) {
    const {svg, map, route, overlay, selected, onPointerDown} = options;
    svg.setAttribute('viewBox', `0 0 ${map.width_m} ${map.height_m}`);
    svg.innerHTML = '';
    svg.append(node('rect', {
      class: 'map-boundary', x: 0, y: 0,
      width: map.width_m, height: map.height_m,
    }));
    for (let x = 1; x < map.width_m; x++) {
      svg.append(node('line', {
        class: 'map-grid-line', x1: x, y1: 0, x2: x, y2: map.height_m,
      }));
    }
    for (let y = 1; y < map.height_m; y++) {
      svg.append(node('line', {
        class: 'map-grid-line', x1: 0, y1: y, x2: map.width_m, y2: y,
      }));
    }
    if (overlay !== 'objects' && route) {
      svg.append(node('path', {
        class: 'planned-route', d: linePath(route.grid_path, map),
      }));
      svg.append(node('path', {
        class: 'return-route', d: linePath(route.return_grid_path, map),
      }));
      (route.waypoints || []).forEach(item => svg.append(node('circle', {
        class: 'route-waypoint', cx: item.east_m,
        cy: map.height_m - item.north_m, r: .28,
      })));
    }
    map.objects.forEach(item => {
      const y = map.height_m - item.north_m;
      if (overlay === 'all') {
        svg.append(node('rect', {
          class: 'safety-envelope',
          x: item.east_m - item.width_m / 2 - 1,
          y: y - item.depth_m / 2 - 1,
          width: item.width_m + 2, height: item.depth_m + 2,
          transform: `rotate(${-item.yaw_deg} ${item.east_m} ${y})`,
        }));
      }
      const shape = node('rect', {
        class: `map-object${item.object_id === selected ? ' selected' : ''}`,
        x: item.east_m - item.width_m / 2,
        y: y - item.depth_m / 2,
        width: item.width_m, height: item.depth_m,
        rx: item.asset_id === 'reactor' || item.asset_id === 'pole' ?
          Math.min(item.width_m, item.depth_m) / 2 : .12,
        fill: colors[item.asset_id],
        transform: `rotate(${-item.yaw_deg} ${item.east_m} ${y})`,
        'data-object': item.object_id,
      });
      shape.addEventListener('pointerdown', event =>
        onPointerDown(event, item.object_id)
      );
      svg.append(shape);
    });
    const startY = map.height_m - map.start_north_m;
    svg.append(node('circle', {
      class: 'map-start', cx: map.start_east_m, cy: startY, r: .35,
    }));
    svg.append(node('line', {
      class: 'map-yaw', x1: map.start_east_m, y1: startY,
      x2: map.start_east_m + Math.sin(map.start_yaw_deg * Math.PI / 180),
      y2: startY - Math.cos(map.start_yaw_deg * Math.PI / 180),
    }));
  }

  root.MapCanvas = {colors, eventPoint, node, render};
})(typeof globalThis === 'undefined' ? this : globalThis);

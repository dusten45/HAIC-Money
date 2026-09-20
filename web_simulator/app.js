(function () {
  "use strict";

  const LEGACY_SCHEMA_VERSION = 1;
  const MAP_SCHEMA_VERSION = 2;
  const RUN_SCHEMA_VERSION = 2;
  const DEFAULT_MAP = {
    schema_version: MAP_SCHEMA_VERSION,
    map_kind: "official",
    map_id: "official-track-1-seed-42",
    track_id: 1,
    seed: 42,
    obstacle_mode: "official_plus_custom",
    obstacles: [],
    max_steps: 2000,
    frame_skip: 4
  };

  const state = {
    mapSpec: { ...DEFAULT_MAP },
    preview: null,
    logs: [],
    runState: null,
    apiAvailable: false,
    manualRunId: null,
    manualTimer: null,
    manualPaused: true,
    manualActionInFlight: false,
    manualKeys: new Set(),
    automaticRunActive: false,
    requestControllers: new Set()
  };

  const $ = (id) => document.getElementById(id);

  function setStatus(message, isError) {
    const target = $("file-status");
    if (target) {
      target.textContent = message;
      target.classList.toggle("error", Boolean(isError));
    }
  }

  function setMapValidation(message, isError) {
    const target = $("map-validation");
    if (!target) return;
    target.textContent = message;
    target.classList.toggle("error", Boolean(isError));
  }

  function setApiStatus(available) {
    state.apiAvailable = Boolean(available);
    const target = $("local-api-status");
    if (!target) return;
    target.textContent = available ? "로컬 실행기 연결됨" : "파일 보기 모드";
    target.classList.toggle("muted", !available);
    if ($("start-agent-run")) setManualControls(Boolean(state.manualRunId));
  }

  async function apiRequest(path, options = {}) {
    const { timeoutMs = 30000, ...fetchOptions } = options;
    const controller = new AbortController();
    state.requestControllers.add(controller);
    const timeout = window.setTimeout(() => controller.abort(), timeoutMs);
    try {
      const headers = new Headers(fetchOptions.headers || {});
      headers.set("Accept", "application/json");
      if (fetchOptions.body && !headers.has("Content-Type")) headers.set("Content-Type", "application/json");
      const response = await fetch(path, { ...fetchOptions, headers, signal: controller.signal });
      const content = await response.text();
      let payload = {};
      if (content) {
        try { payload = JSON.parse(content); } catch (_error) { throw new Error("로컬 API 응답을 JSON으로 읽지 못했습니다."); }
      }
      if (!response.ok) throw new Error(payload.error || `로컬 API 요청이 실패했습니다 (${response.status}).`);
      return payload;
    } finally {
      window.clearTimeout(timeout);
      state.requestControllers.delete(controller);
    }
  }

  async function checkLocalApi() {
    try {
      const response = await apiRequest("/api/health", { timeoutMs: 1800 });
      setApiStatus(response.status === "ok");
    } catch (_error) {
      setApiStatus(false);
    }
    return state.apiAvailable;
  }

  function finiteNumber(value, name, minimum, maximum) {
    const number = Number(value);
    if (!Number.isFinite(number) || number < minimum || number > maximum) {
      throw new Error(`${name}은 ${minimum}에서 ${maximum} 사이여야 합니다.`);
    }
    return number;
  }

  function normalizeObstacles(payload) {
    const obstacles = Array.isArray(payload.obstacles) ? payload.obstacles : [];
    if (obstacles.length > 64) throw new Error("사용자 장애물은 최대 64개입니다.");
    return obstacles.map((obstacle, index) => {
      if (!obstacle || typeof obstacle !== "object") throw new Error(`obstacles[${index}] 형식이 잘못되었습니다.`);
      return {
        progress: finiteNumber(obstacle.progress, "progress", 0, 1),
        lateral: finiteNumber(obstacle.lateral, "lateral", -1, 1),
        radius: finiteNumber(obstacle.radius, "radius", 0.2, 4)
      };
    });
  }

  function normalizeMap(payload) {
    if (!payload || typeof payload !== "object") throw new Error("맵 JSON은 객체여야 합니다.");
    const version = payload.schema_version ?? LEGACY_SCHEMA_VERSION;
    if (![LEGACY_SCHEMA_VERSION, MAP_SCHEMA_VERSION].includes(version)) throw new Error("지원하지 않는 map schema version입니다.");
    const obstacles = normalizeObstacles(payload);
    const kind = payload.map_kind || "official";
    const common = {
      schema_version: MAP_SCHEMA_VERSION,
      obstacles,
      max_steps: Math.trunc(finiteNumber(payload.max_steps ?? 2000, "max_steps", 1, 10000)),
      frame_skip: Math.trunc(finiteNumber(payload.frame_skip ?? 4, "frame_skip", 1, 16))
    };
    if (kind === "custom") {
      if (version !== MAP_SCHEMA_VERSION) throw new Error("자체 맵은 schema version 2가 필요합니다.");
      const mapId = String(payload.map_id || "");
      if (!/^custom-track-[A-Za-z0-9][A-Za-z0-9_-]{0,63}$/.test(mapId)) throw new Error("자체 맵 ID는 custom-track-으로 시작해야 합니다.");
      const rawGeometry = payload.geometry;
      if (!rawGeometry || typeof rawGeometry !== "object" || !Array.isArray(rawGeometry.centerline)) throw new Error("자체 맵 중심선 geometry가 없습니다.");
      const centerline = rawGeometry.centerline.map((point, index) => {
        if (!Array.isArray(point) || point.length !== 2) throw new Error(`centerline[${index}]에는 X, Y 좌표가 필요합니다.`);
        return [
          finiteNumber(point[0], `centerline[${index}].x`, -100000, 100000),
          finiteNumber(point[1], `centerline[${index}].y`, -100000, 100000)
        ];
      });
      if (centerline.length > 256) throw new Error("중심선은 최대 256개 점까지 편집할 수 있습니다.");
      const direction = Number(rawGeometry.direction ?? 1);
      if (![1, -1].includes(direction)) throw new Error("주행 방향은 1 또는 -1이어야 합니다.");
      return {
        ...common,
        map_kind: "custom",
        map_id: mapId,
        geometry: {
          centerline,
          width: finiteNumber(rawGeometry.width, "width", 0.5, 100),
          start_index: Math.trunc(finiteNumber(rawGeometry.start_index ?? 0, "start_index", 0, Math.max(0, centerline.length - 1))),
          direction
        },
        obstacle_mode: "custom_only",
        generator: payload.generator && typeof payload.generator === "object" ? payload.generator : {},
        metadata: payload.metadata && typeof payload.metadata === "object" ? payload.metadata : {}
      };
    }
    if (kind !== "official") throw new Error("map_kind은 official 또는 custom이어야 합니다.");
    const mode = payload.obstacle_mode || "official";
    if (!["official", "custom_only", "official_plus_custom"].includes(mode)) throw new Error("장애물 모드가 잘못되었습니다.");
    const trackId = Math.trunc(finiteNumber(payload.track_id, "track_id", 1, Number.MAX_SAFE_INTEGER));
    const seed = Math.trunc(finiteNumber(payload.seed, "seed", 0, 4294967295));
    return {
      ...common,
      map_kind: "official",
      map_id: payload.map_id || `official-track-${trackId}-seed-${seed}`,
      track_id: trackId,
      seed,
      obstacle_mode: mode,
      metadata: payload.metadata && typeof payload.metadata === "object" ? payload.metadata : {}
    };
  }

  function readForm() {
    const common = {
      ...state.mapSpec,
      max_steps: $("max-steps").value,
      frame_skip: $("frame-skip").value,
      obstacles: state.mapSpec.obstacles
    };
    if ($("map-kind").value === "custom") {
      if (state.mapSpec.map_kind !== "custom") throw new Error("먼저 자체 트랙을 생성하거나 불러오세요.");
      state.mapSpec = normalizeMap({
        ...common,
        map_kind: "custom",
        map_id: $("custom-map-id").value.trim(),
        geometry: {
          ...state.mapSpec.geometry,
          width: $("custom-width").value
        }
      });
    } else {
      const trackId = Math.trunc(finiteNumber($("track-id").value, "track_id", 1, Number.MAX_SAFE_INTEGER));
      const seed = Math.trunc(finiteNumber($("seed").value, "seed", 0, 4294967295));
      state.mapSpec = normalizeMap({
        ...common,
        map_kind: "official",
        map_id: `official-track-${trackId}-seed-${seed}`,
        track_id: trackId,
        seed,
        obstacle_mode: $("obstacle-mode").value
      });
    }
    return state.mapSpec;
  }

  function syncForm() {
    const map = state.mapSpec;
    $("map-kind").value = map.map_kind || "official";
    if (map.map_kind === "custom") {
      $("custom-map-id").value = map.map_id;
      $("custom-width").value = map.geometry.width;
      const generator = map.generator || {};
      $("design-seed").value = generator.design_seed ?? 42;
      $("custom-template").value = generator.template || "oval";
    } else {
      $("track-id").value = map.track_id;
      $("seed").value = map.seed;
      $("obstacle-mode").value = map.obstacle_mode;
    }
    $("max-steps").value = map.max_steps;
    $("frame-skip").value = map.frame_skip;
    updateMapKindVisibility();
  }

  function updateMapKindVisibility() {
    const custom = $("map-kind") && $("map-kind").value === "custom";
    $("official-map-fields").hidden = custom;
    $("custom-builder").hidden = !custom;
    $("control-point-editor").hidden = !custom;
  }

  function previewFromCustomMap(map) {
    if (!map || map.map_kind !== "custom" || !map.geometry || !map.geometry.centerline.length) return null;
    const centerline = map.geometry.centerline;
    const points = centerline.map(([x, y], index) => {
      const previous = centerline[(index - 1 + centerline.length) % centerline.length];
      const following = centerline[(index + 1) % centerline.length];
      const tangent = Math.atan2(following[1] - previous[1], following[0] - previous[0]);
      return [index / centerline.length, tangent - Math.PI / 2, x, y];
    });
    return {
      track: { points, width: map.geometry.width },
      official_obstacles: [],
      custom_obstacles: []
    };
  }

  function crossProduct(a, b, c) {
    return (b[0] - a[0]) * (c[1] - a[1]) - (b[1] - a[1]) * (c[0] - a[0]);
  }

  function segmentsIntersect(a, b, c, d) {
    const epsilon = 1e-9;
    const onSegment = (first, second, point) =>
      point[0] >= Math.min(first[0], second[0]) - epsilon && point[0] <= Math.max(first[0], second[0]) + epsilon &&
      point[1] >= Math.min(first[1], second[1]) - epsilon && point[1] <= Math.max(first[1], second[1]) + epsilon;
    const abC = crossProduct(a, b, c);
    const abD = crossProduct(a, b, d);
    const cdA = crossProduct(c, d, a);
    const cdB = crossProduct(c, d, b);
    if (((abC > epsilon && abD < -epsilon) || (abC < -epsilon && abD > epsilon)) &&
        ((cdA > epsilon && cdB < -epsilon) || (cdA < -epsilon && cdB > epsilon))) return true;
    return (Math.abs(abC) <= epsilon && onSegment(a, b, c)) ||
      (Math.abs(abD) <= epsilon && onSegment(a, b, d)) ||
      (Math.abs(cdA) <= epsilon && onSegment(c, d, a)) ||
      (Math.abs(cdB) <= epsilon && onSegment(c, d, b));
  }

  function customMapValidation(map = state.mapSpec) {
    if (!map || map.map_kind !== "custom") return "";
    const geometry = map.geometry;
    if (!geometry || !Array.isArray(geometry.centerline)) return "중심선이 없습니다. 트랙을 생성하거나 다시 불러오세요.";
    const points = geometry.centerline;
    if (points.length < 12) return "중심선은 최소 12개 점이 필요합니다.";
    if (!Number.isFinite(geometry.width) || geometry.width < 0.5 || geometry.width > 100) return "도로 반폭은 0.5에서 100 사이여야 합니다.";
    const lengths = points.map((point, index) => Math.hypot(
      points[(index + 1) % points.length][0] - point[0],
      points[(index + 1) % points.length][1] - point[1]
    ));
    if (lengths.some((length) => !Number.isFinite(length) || length <= 1e-6)) return "중심선에 중복되거나 잘못된 점이 있습니다.";
    if (lengths.reduce((sum, length) => sum + length, 0) < geometry.width * 8) return "도로 폭에 비해 중심선이 너무 짧습니다.";
    for (let first = 0; first < points.length; first += 1) {
      for (let second = first + 1; second < points.length; second += 1) {
        if (first === second || (first + 1) % points.length === second || (second + 1) % points.length === first) continue;
        if (segmentsIntersect(points[first], points[(first + 1) % points.length], points[second], points[(second + 1) % points.length])) {
          return `중심선이 ${first}번과 ${second}번 구간에서 교차합니다.`;
        }
      }
    }
    for (let index = 0; index < points.length; index += 1) {
      const previous = points[(index - 1 + points.length) % points.length];
      const point = points[index];
      const following = points[(index + 1) % points.length];
      const incoming = [point[0] - previous[0], point[1] - previous[1]];
      const outgoing = [following[0] - point[0], following[1] - point[1]];
      const dot = incoming[0] * outgoing[0] + incoming[1] * outgoing[1];
      const cosine = dot / (lengths[(index - 1 + lengths.length) % lengths.length] * lengths[index]);
      if (cosine < -0.995) return `${index}번 점의 회전이 너무 급합니다.`;
    }
    if (!/^custom-track-[A-Za-z0-9][A-Za-z0-9_-]{0,63}$/.test(map.map_id || "")) return "맵 ID는 custom-track-으로 시작해야 합니다.";
    return "";
  }

  function renderControlPointTable() {
    const table = $("control-point-table");
    const map = state.mapSpec;
    const points = map.map_kind === "custom" && map.geometry ? map.geometry.centerline : [];
    $("add-control-point").disabled = points.length < 1 || points.length >= 256;
    $("remove-control-point").disabled = points.length <= 12;
    if (!points.length) {
      table.innerHTML = '<tr><td colspan="3" class="empty-cell">트랙을 생성하면 중심선을 편집할 수 있습니다.</td></tr>';
      return;
    }
    table.innerHTML = points.map((point, index) => `
      <tr>
        <td>${index + 1}</td>
        <td><input data-point-index="${index}" data-axis="0" type="number" step="0.1" value="${Number(point[0].toFixed(3))}"></td>
        <td><input data-point-index="${index}" data-axis="1" type="number" step="0.1" value="${Number(point[1].toFixed(3))}"></td>
      </tr>`).join("");
  }

  function refreshCustomMapPreview() {
    state.preview = previewFromCustomMap(state.mapSpec);
    drawTrack($("map-canvas"), state.preview, state.mapSpec.obstacles);
    const validation = customMapValidation(state.mapSpec);
    setMapValidation(validation || "자체 트랙 geometry가 유효합니다.", Boolean(validation));
  }

  function editControlPoint(event) {
    const input = event.target.closest("[data-point-index]");
    if (!input || state.mapSpec.map_kind !== "custom") return;
    const value = Number(input.value);
    if (!Number.isFinite(value) || Math.abs(value) > 100000) {
      setMapValidation("중심선 좌표는 -100000에서 100000 사이의 유한한 값이어야 합니다.", true);
      return;
    }
    const index = Number(input.dataset.pointIndex);
    const axis = Number(input.dataset.axis);
    state.mapSpec.geometry.centerline[index][axis] = value;
    refreshCustomMapPreview();
  }

  function addControlPoint() {
    if (state.mapSpec.map_kind !== "custom") return;
    const points = state.mapSpec.geometry.centerline;
    if (points.length >= 256) throw new Error("중심선은 최대 256개 점까지 편집할 수 있습니다.");
    if (!points.length) points.push([0, 0]);
    else {
      const first = points[0];
      const last = points[points.length - 1];
      points.push([(first[0] + last[0]) / 2, (first[1] + last[1]) / 2]);
    }
    renderMap();
  }

  function removeControlPoint() {
    if (state.mapSpec.map_kind !== "custom") return;
    const points = state.mapSpec.geometry.centerline;
    if (points.length <= 12) throw new Error("유효성 검사를 위해 중심선 점을 최소 12개 유지해야 합니다.");
    points.pop();
    renderMap();
  }

  function worldBounds(points) {
    const xs = points.map((point) => point[2]);
    const ys = points.map((point) => point[3]);
    if (!xs.length || !ys.length) return { minX: -1, maxX: 1, minY: -1, maxY: 1 };
    const padding = 8;
    return {
      minX: Math.min(...xs) - padding,
      maxX: Math.max(...xs) + padding,
      minY: Math.min(...ys) - padding,
      maxY: Math.max(...ys) + padding
    };
  }

  function project(point, bounds, width, height) {
    const x = point[0];
    const y = point[1];
    const scale = Math.min(width / (bounds.maxX - bounds.minX), height / (bounds.maxY - bounds.minY));
    return {
      x: (x - bounds.minX) * scale + (width - (bounds.maxX - bounds.minX) * scale) / 2,
      y: height - ((y - bounds.minY) * scale + (height - (bounds.maxY - bounds.minY) * scale) / 2)
    };
  }

  function drawTrack(canvas, preview, obstacles, emptyId = "canvas-empty") {
    const context = canvas.getContext("2d");
    const width = canvas.width;
    const height = canvas.height;
    context.clearRect(0, 0, width, height);
    context.fillStyle = "#080b10";
    context.fillRect(0, 0, width, height);
    const empty = emptyId ? $(emptyId) : null;
    if (!preview || !preview.track || !Array.isArray(preview.track.points) || !preview.track.points.length) {
      if (empty) empty.classList.remove("hidden");
      return;
    }
    if (empty) empty.classList.add("hidden");
    const points = preview.track.points;
    const bounds = worldBounds(points);
    const scale = Math.min(width / (bounds.maxX - bounds.minX), height / (bounds.maxY - bounds.minY));
    const mapped = points.map((point) => project([point[2], point[3]], bounds, width, height));
    context.lineJoin = "round";
    context.lineCap = "round";
    context.beginPath();
    mapped.forEach((point, index) => index ? context.lineTo(point.x, point.y) : context.moveTo(point.x, point.y));
    context.closePath();
    context.strokeStyle = "#394556";
    context.lineWidth = Math.max(18, Math.min(150, preview.track.width * 2 * scale));
    context.stroke();
    const edgeLines = [-1, 1].map((side) => points.map((point) => {
      const beta = point[1];
      return project([
        point[2] + side * preview.track.width * Math.cos(beta),
        point[3] + side * preview.track.width * Math.sin(beta)
      ], bounds, width, height);
    }));
    edgeLines.forEach((edge) => {
      context.beginPath();
      edge.forEach((point, index) => index ? context.lineTo(point.x, point.y) : context.moveTo(point.x, point.y));
      context.closePath();
      context.strokeStyle = "#c1ccd566";
      context.lineWidth = 2;
      context.stroke();
    });
    context.beginPath();
    mapped.forEach((point, index) => index ? context.lineTo(point.x, point.y) : context.moveTo(point.x, point.y));
    context.closePath();
    context.strokeStyle = "#8c98a9";
    context.lineWidth = 2;
    context.setLineDash([8, 8]);
    context.stroke();
    context.setLineDash([]);
    const drawObstacle = (position, radius, color) => {
      const point = project(position, bounds, width, height);
      context.beginPath();
      context.arc(point.x, point.y, Math.max(5, radius * scale), 0, Math.PI * 2);
      context.fillStyle = color;
      context.fill();
      context.strokeStyle = "#ffffff55";
      context.lineWidth = 1;
      context.stroke();
    };
    (preview.official_obstacles || []).forEach((position) => drawObstacle(position, 1.2, "#f2c66d"));
    (obstacles || []).forEach((obstacle) => {
      const index = Math.min(points.length - 1, Math.max(0, Math.round(obstacle.progress * (points.length - 1))));
      const trackPoint = points[index];
      const beta = trackPoint[1];
      const offset = obstacle.lateral * Math.min(preview.track.width * .6, preview.track.width - obstacle.radius);
      drawObstacle([trackPoint[2] + offset * Math.cos(beta), trackPoint[3] + offset * Math.sin(beta)], obstacle.radius, "#7ce2b2");
    });
    const start = mapped[0];
    context.beginPath();
    context.arc(start.x, start.y, 5, 0, Math.PI * 2);
    context.fillStyle = "#8eb5ff";
    context.fill();
  }

  function renderObstacleTable() {
    const table = $("obstacle-table");
    if (!state.mapSpec.obstacles.length) {
      table.innerHTML = '<tr><td colspan="5" class="empty-cell">아직 사용자 장애물이 없습니다.</td></tr>';
      return;
    }
    table.innerHTML = state.mapSpec.obstacles.map((obstacle, index) => `
      <tr>
        <td>${index + 1}</td>
        <td><input data-obstacle="${index}" data-key="progress" type="number" min="0" max="1" step="0.01" value="${obstacle.progress}"></td>
        <td><input data-obstacle="${index}" data-key="lateral" type="number" min="-1" max="1" step="0.01" value="${obstacle.lateral}"></td>
        <td><input data-obstacle="${index}" data-key="radius" type="number" min="0.2" max="4" step="0.1" value="${obstacle.radius}"></td>
        <td><button type="button" class="button ghost delete-obstacle" data-index="${index}">삭제</button></td>
      </tr>`).join("");
  }

  function renderMap() {
    renderObstacleTable();
    renderControlPointTable();
    drawTrack($("map-canvas"), state.preview, state.mapSpec.obstacles);
    const map = state.mapSpec;
    $("map-status").textContent = map.map_kind === "custom"
      ? `${map.map_id} · ${map.obstacles.length}개 장애물`
      : `공식 ${map.track_id} · seed ${map.seed} · ${map.obstacles.length}개 장애물`;
    const validation = customMapValidation(map);
    setMapValidation(validation || (map.map_kind === "custom" ? "자체 트랙 geometry가 유효합니다." : "공식 맵 설정입니다."), Boolean(validation));
  }

  function normalizeRun(payload) {
    if (!payload || typeof payload !== "object") throw new Error("실행 로그는 객체여야 합니다.");
    const version = payload.schema_version ?? LEGACY_SCHEMA_VERSION;
    if (![LEGACY_SCHEMA_VERSION, RUN_SCHEMA_VERSION].includes(version)) throw new Error("지원하지 않는 run schema version입니다.");
    if (!payload.track || !Array.isArray(payload.track.points) || !payload.track.points.length) {
      throw new Error("run.json에 track.points가 없습니다.");
    }
    if (!Array.isArray(payload.steps)) throw new Error("run.json에 steps가 없습니다.");
    const steps = payload.steps.map((step, index) => {
      if (!step || typeof step !== "object") throw new Error(`steps[${index}] 형식이 잘못되었습니다.`);
      if (!Array.isArray(step.position) || step.position.length !== 2) throw new Error(`steps[${index}].position이 잘못되었습니다.`);
      return {
        ...step,
        step: Number(step.step ?? index),
        sim_time_s: Number(step.sim_time_s ?? index),
        position: [Number(step.position[0]), Number(step.position[1])],
        angle: Number(step.angle || 0),
        progress: Number(step.progress || 0),
        damage: Number(step.damage || 0),
        collision: Boolean(step.collision),
        terminated: Boolean(step.terminated),
        truncated: Boolean(step.truncated)
      };
    });
    return {
      ...payload,
      run: payload.run && typeof payload.run === "object" ? payload.run : {},
      track: {
        ...payload.track,
        width: Number(payload.track.width || 80),
        points: payload.track.points.map((point) => point.map(Number))
      },
      steps,
      summary: payload.summary && typeof payload.summary === "object" ? payload.summary : {},
      frames: Array.isArray(payload.frames) ? payload.frames.slice() : null
    };
  }

  function loadRunLog(payload) {
    return {
      log: normalizeRun(payload),
      stepIndex: 0,
      running: false,
      speed: Number($("speed") ? $("speed").value : 1) || 1
    };
  }

  function formatLapTime(value) {
    if (value === null || value === undefined || !Number.isFinite(Number(value))) return "—";
    return `${(Number(value) / 1000).toFixed(3)} s`;
  }

  function formatPercent(value) {
    if (value === null || value === undefined || !Number.isFinite(Number(value))) return "—";
    return `${(Number(value) * 100).toFixed(1)}%`;
  }

  function logMap(log) {
    return log && log.run && log.run.map && typeof log.run.map === "object" ? log.run.map : null;
  }

  function replayPreview(log) {
    const map = logMap(log);
    const currentMapMatches = map && state.mapSpec && map.map_id && map.map_id === state.mapSpec.map_id;
    const source = map && map.map_kind === "custom"
      ? previewFromCustomMap(normalizeMap(map))
      : (currentMapMatches ? state.preview : null) || {};
    return {
      track: log.track,
      official_obstacles: source.official_obstacles || [],
      custom_obstacles: source.custom_obstacles || [],
      map
    };
  }

  function replayObstacles(log) {
    const map = logMap(log);
    if (map && Array.isArray(map.obstacles)) return map.obstacles;
    return [];
  }

  function drawReplay(log, stepIndex) {
    const preview = replayPreview(log);
    drawTrack($("replay-canvas"), preview, replayObstacles(log), "replay-empty");
    const points = log.track.points;
    const bounds = worldBounds(points);
    const canvas = $("replay-canvas");
    const context = canvas.getContext("2d");
    const current = log.steps[stepIndex];
    if (!current) return;
    const trajectory = log.steps.slice(0, stepIndex + 1).map((step) => project(step.position, bounds, canvas.width, canvas.height));
    context.beginPath();
    trajectory.forEach((point, index) => index ? context.lineTo(point.x, point.y) : context.moveTo(point.x, point.y));
    context.strokeStyle = "#8eb5ff";
    context.lineWidth = 3;
    context.lineCap = "round";
    context.stroke();
    const car = project(current.position, bounds, canvas.width, canvas.height);
    const scale = Math.min(canvas.width / (bounds.maxX - bounds.minX), canvas.height / (bounds.maxY - bounds.minY));
    context.save();
    context.translate(car.x, car.y);
    context.rotate(-current.angle);
    context.beginPath();
    context.moveTo(12, 0);
    context.lineTo(-9, -7);
    context.lineTo(-6, 7);
    context.closePath();
    context.fillStyle = "#ffffff";
    context.fill();
    context.strokeStyle = "#8eb5ff";
    context.lineWidth = 2;
    context.stroke();
    context.restore();
    context.beginPath();
    context.arc(car.x, car.y, Math.max(4, scale * .35), 0, Math.PI * 2);
    context.fillStyle = "#7ce2b2";
    context.fill();
    const camera = $("camera-frame");
    const frame = log.frames && log.frames[stepIndex];
    if (frame) {
      camera.src = `data:image/jpeg;base64,${frame}`;
      camera.hidden = false;
    } else {
      camera.hidden = true;
      camera.removeAttribute("src");
    }
  }

  function updateMetrics(runState) {
    if (!runState) return;
    const log = runState.log;
    const summary = log.summary || {};
    const map = logMap(log) || {};
    const current = log.steps[runState.stepIndex];
    $("metric-result").textContent = summary.finished ? "FINISHED" : "DNF";
    $("metric-result").style.color = summary.finished ? "var(--accent)" : "var(--warning)";
    $("metric-lap-time").textContent = formatLapTime(summary.lap_time_ms);
    $("metric-progress").textContent = formatPercent(current ? current.progress : summary.progress);
    $("metric-damage").textContent = formatPercent(current ? current.damage : summary.damage);
    $("metric-collisions").textContent = String(summary.collision_count ?? "—");
    $("metric-track").textContent = map.map_kind === "custom"
      ? (map.map_id || log.run.map_ref?.map_id || "custom")
      : `${map.track_id ?? "—"} / ${map.seed ?? "—"}`;
    $("current-step").textContent = current ? String(runState.stepIndex + 1) : "0";
    $("total-steps").textContent = String(log.steps.length);
    $("timeline").max = String(Math.max(0, log.steps.length - 1));
    $("timeline").value = String(runState.stepIndex);
    $("run-status").textContent = runState.running ? "재생 중" : "로그 로드됨";
    $("run-status").classList.toggle("muted", !runState.running);
  }

  function renderRunFrame(runState, stepIndex) {
    if (!runState || !runState.log.steps.length) return;
    const last = runState.log.steps.length - 1;
    runState.stepIndex = Math.max(0, Math.min(last, Math.trunc(Number(stepIndex))));
    drawReplay(runState.log, runState.stepIndex);
    updateMetrics(runState);
  }

  function summarizeRuns(runs) {
    return runs.map((item, index) => {
      const log = item && item.log ? item.log : item;
      const map = logMap(log) || {};
      const summary = log.summary || {};
      return {
        run_id: log.run && log.run.run_id ? String(log.run.run_id) : `run-${index + 1}`,
        finished: Boolean(summary.finished),
        lap_time_ms: summary.lap_time_ms === null || summary.lap_time_ms === undefined ? null : Number(summary.lap_time_ms),
        progress: Number(summary.progress || 0),
        damage: Number(summary.damage || 0),
        collision_count: Number(summary.collision_count || 0),
        seed: map.seed,
        track_id: map.track_id
      };
    }).sort((left, right) => {
      if (left.finished !== right.finished) return left.finished ? -1 : 1;
      const leftLap = left.lap_time_ms === null ? Number.POSITIVE_INFINITY : left.lap_time_ms;
      const rightLap = right.lap_time_ms === null ? Number.POSITIVE_INFINITY : right.lap_time_ms;
      if (leftLap !== rightLap) return leftLap - rightLap;
      return right.progress - left.progress;
    });
  }

  function renderComparison() {
    const rows = summarizeRuns(state.logs);
    const table = $("run-table");
    if (!rows.length) {
      table.innerHTML = '<tr><td colspan="6" class="empty-cell">로그를 추가하면 기록이 쌓입니다.</td></tr>';
      return;
    }
    table.innerHTML = rows.map((row) => `
      <tr>
        <td title="${row.run_id}">${row.run_id.slice(0, 10)}</td>
        <td class="${row.finished ? "finished" : "dnf"}">${row.finished ? "FINISHED" : "DNF"}</td>
        <td>${formatLapTime(row.lap_time_ms)}</td>
        <td>${formatPercent(row.progress)}</td>
        <td>${formatPercent(row.damage)}</td>
        <td>${row.collision_count}</td>
      </tr>`).join("");
  }

  let animationId = null;
  let lastAnimationTime = null;

  function setPlaybackRunning(running) {
    if (!state.runState || !state.runState.log.steps.length) return;
    if (running && state.runState.stepIndex >= state.runState.log.steps.length - 1) {
      state.runState.stepIndex = 0;
      renderRunFrame(state.runState, 0);
    }
    state.runState.running = Boolean(running);
    if (!running) {
      if (animationId !== null) cancelAnimationFrame(animationId);
      animationId = null;
      lastAnimationTime = null;
      updateMetrics(state.runState);
      return;
    }
    if (animationId === null) animationId = requestAnimationFrame(playbackTick);
    updateMetrics(state.runState);
  }

  function advancePlayback() {
    if (!state.runState || !state.runState.log.steps.length) return;
    const next = state.runState.stepIndex + 1;
    if (next >= state.runState.log.steps.length) {
      setPlaybackRunning(false);
      renderRunFrame(state.runState, state.runState.log.steps.length - 1);
      return;
    }
    renderRunFrame(state.runState, next);
  }

  function playbackTick(timestamp) {
    animationId = null;
    if (!state.runState || !state.runState.running) return;
    if (lastAnimationTime === null) lastAnimationTime = timestamp;
    const current = state.runState.log.steps[state.runState.stepIndex];
    const next = state.runState.log.steps[state.runState.stepIndex + 1];
    const stepDelta = next ? Math.max(16, (Number(next.sim_time_s) - Number(current.sim_time_s)) * 1000) : 16;
    const interval = stepDelta / Math.max(.1, state.runState.speed);
    if (timestamp - lastAnimationTime >= interval) {
      lastAnimationTime = timestamp;
      advancePlayback();
    }
    if (state.runState && state.runState.running) animationId = requestAnimationFrame(playbackTick);
  }

  function setRunControls(enabled) {
    ["pause-button", "step-back", "step-forward", "restart-button", "speed", "timeline"].forEach((id) => { $(id).disabled = !enabled; });
    $("play-button").disabled = !enabled;
  }

  function addRunPayload(payload, source) {
    const runState = loadRunLog(payload);
    state.logs.push(runState.log);
    state.runState = runState;
    setRunControls(true);
    renderRunFrame(runState, 0);
    renderComparison();
    setStatus(`${source}을 불러왔습니다. ${runState.log.steps.length}개 스텝`, false);
  }

  function applyMap(payload, source) {
    state.mapSpec = normalizeMap(payload);
    state.preview = payload.preview || previewFromCustomMap(state.mapSpec);
    syncForm();
    renderMap();
    setStatus(`${source}을 불러왔습니다.`, false);
  }

  function applyCustomMap(payload) {
    const map = normalizeMap(payload);
    if (map.map_kind !== "custom") throw new Error("자체 맵 payload가 아닙니다.");
    applyMap({ ...payload, ...map }, map.map_id);
    return state.mapSpec;
  }

  function addObstacle(obstacle) {
    if (state.mapSpec.obstacles.length >= 64) throw new Error("사용자 장애물은 최대 64개입니다.");
    state.mapSpec.obstacles = [...state.mapSpec.obstacles, obstacle];
    renderMap();
  }

  function addObstacleFromForm() {
    readForm();
    addObstacle({
      progress: finiteNumber($("obstacle-progress").value, "progress", 0, 1),
      lateral: finiteNumber($("obstacle-lateral").value, "lateral", -1, 1),
      radius: finiteNumber($("obstacle-radius").value, "radius", 0.2, 4)
    });
    setStatus("사용자 장애물을 추가했습니다.", false);
  }

  function seededRandom(seed) {
    let value = (seed >>> 0) || 1;
    return () => {
      value = (value * 1664525 + 1013904223) >>> 0;
      return value / 4294967296;
    };
  }

  function browserGeneratedCenterline(template, seed) {
    const random = seededRandom(seed);
    if (template === "hairpin") {
      const radius = 22 + random() * 4 - 2;
      const length = 92 + random() * 10 - 5;
      const lineCount = 12;
      const arcCount = 12;
      const points = [];
      for (let index = 0; index < lineCount; index += 1) {
        const ratio = index / lineCount;
        points.push([-length / 2 + length * ratio, radius]);
      }
      for (let index = 0; index < arcCount; index += 1) {
        const angle = Math.PI / 2 - Math.PI * index / arcCount;
        points.push([length / 2 + radius * Math.cos(angle), radius * Math.sin(angle)]);
      }
      for (let index = 0; index < lineCount; index += 1) {
        const ratio = index / lineCount;
        points.push([length / 2 - length * ratio, -radius]);
      }
      for (let index = 0; index < arcCount; index += 1) {
        const angle = -Math.PI / 2 - Math.PI * index / arcCount;
        points.push([-length / 2 + radius * Math.cos(angle), radius * Math.sin(angle)]);
      }
      return points;
    }
    const settings = {
      oval: [64 + random() * 12 - 6, 38 + random() * 8 - 4, 2, 0.025],
      s_curve: [62 + random() * 10 - 5, 42 + random() * 8 - 4, 1, 0.07],
      chicane: [60 + random() * 10 - 5, 40 + random() * 8 - 4, 3, 0.055]
    }[template];
    if (!settings) throw new Error("트랙 템플릿을 확인해주세요.");
    const [radiusX, radiusY, harmonic, amplitude] = settings;
    const phase = random() * 0.36 - 0.18;
    return Array.from({ length: 48 }, (_item, index) => {
      const angle = 2 * Math.PI * index / 48;
      const radius = 1 + amplitude * Math.sin(harmonic * angle + phase);
      return [radiusX * radius * Math.cos(angle), radiusY * radius * Math.sin(angle)];
    });
  }

  function makeBrowserCustomMap(options) {
    const designSeed = Math.trunc(finiteNumber(options.design_seed, "디자인 seed", 0, 4294967295));
    return {
      schema_version: MAP_SCHEMA_VERSION,
      map_id: String(options.map_id),
      map_kind: "custom",
      geometry: {
        centerline: browserGeneratedCenterline(options.template, designSeed),
        width: finiteNumber(options.width, "도로 반폭", 0.5, 100),
        start_index: 0,
        direction: 1
      },
      obstacle_mode: "custom_only",
      obstacles: Array.isArray(options.obstacles) ? options.obstacles : [],
      max_steps: Math.trunc(finiteNumber(options.max_steps, "max_steps", 1, 10000)),
      frame_skip: Math.trunc(finiteNumber(options.frame_skip, "frame_skip", 1, 16)),
      generator: { template: options.template, design_seed: designSeed, client_fallback: true }
    };
  }

  async function generateCustomMap() {
    const designSeed = Math.trunc(finiteNumber($("design-seed").value, "디자인 seed", 0, 4294967295));
    const mapId = $("custom-map-id").value.trim();
    const template = $("custom-template").value;
    const width = finiteNumber($("custom-width").value, "도로 반폭", 0.5, 100);
    if (!/^custom-track-[A-Za-z0-9][A-Za-z0-9_-]{0,63}$/.test(mapId)) throw new Error("자체 맵 ID는 custom-track-으로 시작해야 합니다.");
    const payload = {
      map_kind: "custom",
      map_id: mapId,
      design_seed: designSeed,
      template,
      width,
      max_steps: Number($("max-steps").value),
      frame_skip: Number($("frame-skip").value),
      obstacles: state.mapSpec.map_kind === "custom" ? state.mapSpec.obstacles : []
    };
    let result;
    if (state.apiAvailable) {
      result = await apiRequest("/api/maps/generate", {
        method: "POST",
        body: JSON.stringify(payload)
      });
    } else {
      result = makeBrowserCustomMap(payload);
    }
    applyCustomMap(result);
    setStatus(`${mapId} 트랙을 생성했습니다.${state.apiAvailable ? "" : " 로컬 실행기는 파일 전용 모드입니다."}`, false);
  }

  function generateRandomObstacles() {
    readForm();
    const count = Math.trunc(finiteNumber($("random-count").value, "개수", 1, 64));
    const seed = state.mapSpec.map_kind === "custom"
      ? Number(state.mapSpec.generator.design_seed || 0)
      : state.mapSpec.seed;
    const random = seededRandom(seed ^ 0xc0de);
    const generated = Array.from({ length: count }, (_, index) => ({
      progress: Number((0.12 + (index + 1) / (count + 1) * 0.76).toFixed(4)),
      lateral: Number((random() * 1.1 - 0.55).toFixed(4)),
      radius: 1.2
    }));
    state.mapSpec.obstacles = generated.slice(0, 64);
    if (state.mapSpec.obstacle_mode === "official") state.mapSpec.obstacle_mode = "official_plus_custom";
    syncForm();
    renderMap();
    setStatus(`${generated.length}개 장애물을 seed 기반으로 생성했습니다.`, false);
  }

  function downloadMap() {
    const map = readForm();
    const validation = customMapValidation(map);
    if (validation) throw new Error(validation);
    const payload = { ...map, ...(state.preview ? { preview: state.preview } : {}) };
    const blob = new Blob([JSON.stringify(payload, null, 2)], { type: "application/json" });
    const link = document.createElement("a");
    link.href = URL.createObjectURL(blob);
    link.download = "map.json";
    link.click();
    URL.revokeObjectURL(link.href);
    setStatus("편집한 map.json을 다운로드했습니다.", false);
  }

  async function saveMap() {
    const map = readForm();
    const validation = customMapValidation(map);
    if (validation) throw new Error(validation);
    if (!state.apiAvailable) {
      downloadMap();
      setStatus("로컬 실행기가 없어 JSON 파일로 저장했습니다.", false);
      return;
    }
    const response = await apiRequest("/api/maps/save", {
      method: "POST",
      body: JSON.stringify({ map })
    });
    applyMap(response, map.map_id);
    setStatus(`${map.map_id} 맵을 로컬 maps 폴더에 저장했습니다.`, false);
  }

  function readJsonFile(file, callback) {
    const reader = new FileReader();
    reader.onload = () => {
      try { callback(JSON.parse(reader.result)); } catch (error) { setStatus(`JSON을 읽지 못했습니다: ${error.message}`, true); }
    };
    reader.onerror = () => setStatus("파일을 읽지 못했습니다.", true);
    reader.readAsText(file);
  }

  const MANUAL_ACTION_INTERVAL_MS = 100;
  const MANUAL_KEYS = new Set(["ArrowLeft", "ArrowRight", "ArrowUp", "Space"]);

  function setLiveRunStatus(message, isError = false) {
    const target = $("live-run-status");
    target.textContent = message;
    target.classList.toggle("error", Boolean(isError));
  }

  function setManualControls(active) {
    const manual = Boolean(active && !state.automaticRunActive);
    const unavailable = !state.apiAvailable;
    $("start-manual-run").disabled = unavailable || manual || state.automaticRunActive;
    $("start-agent-run").disabled = unavailable || manual || state.automaticRunActive;
    $("run-policy").disabled = unavailable || manual || state.automaticRunActive;
    $("record-frames").disabled = unavailable || manual || state.automaticRunActive;
    $("pause-manual-run").disabled = !manual;
    $("step-manual-run").disabled = !manual;
    $("finish-manual-run").disabled = !manual;
    $("pause-manual-run").textContent = state.manualPaused ? "재개" : "일시정지";
  }

  function currentManualAction() {
    return manualActionFromKeys(state.manualKeys);
  }

  function manualActionFromKeys(keys) {
    const left = keys.has("ArrowLeft");
    const right = keys.has("ArrowRight");
    const steer = left === right ? 0 : (left ? -1 : 1);
    return [steer, keys.has("ArrowUp") ? 1 : 0, keys.has("Space") ? 1 : 0];
  }

  function updateManualActionStatus() {
    const [steer, gas, brake] = currentManualAction();
    const direction = steer < 0 ? "좌회전" : (steer > 0 ? "우회전" : "직진");
    const inputs = [direction, gas ? "가속" : "", brake ? "제동" : ""].filter(Boolean);
    $("manual-action-status").textContent = inputs.join(" · ");
  }

  function stopManualTimer() {
    if (state.manualTimer !== null) window.clearTimeout(state.manualTimer);
    state.manualTimer = null;
  }

  function scheduleManualAction() {
    if (!state.manualRunId || state.manualPaused || state.automaticRunActive || document.hidden || state.manualTimer !== null) return;
    state.manualTimer = window.setTimeout(async () => {
      state.manualTimer = null;
      try { await sendManualAction(currentManualAction()); } catch (_error) { return; }
      scheduleManualAction();
    }, MANUAL_ACTION_INTERVAL_MS);
  }

  async function sendManualAction(action) {
    if (!state.manualRunId) throw new Error("먼저 수동 주행을 시작하세요.");
    if (state.manualActionInFlight) return null;
    state.manualActionInFlight = true;
    try {
      const response = await apiRequest(`/api/runs/${encodeURIComponent(state.manualRunId)}/action`, {
        method: "POST",
        body: JSON.stringify({ action }),
        timeoutMs: 15000
      });
      const step = response.step || {};
      setLiveRunStatus(`스텝 ${Number(step.step || 0) + 1} · 진행 ${formatPercent(step.progress)}`);
      if (response.done) {
        state.manualPaused = true;
        stopManualTimer();
        await finishRun("completed");
      }
      return response;
    } catch (error) {
      state.manualPaused = true;
      stopManualTimer();
      setManualControls(Boolean(state.manualRunId));
      setLiveRunStatus(`실행 오류: ${error.message}`, true);
      setStatus(`수동 실행 요청이 중단되었습니다: ${error.message}`, true);
      throw error;
    } finally {
      state.manualActionInFlight = false;
    }
  }

  async function startRun(policy) {
    if (!state.apiAvailable) throw new Error("로컬 실행기가 연결되지 않았습니다. 통합 실행기를 먼저 켜주세요.");
    if (state.manualRunId || state.automaticRunActive) throw new Error("이미 실행 중인 주행이 있습니다.");
    const map = readForm();
    const validation = customMapValidation(map);
    if (validation) throw new Error(validation);
    const recordFrames = $("record-frames").checked;
    if (policy === "manual") {
      const started = await apiRequest("/api/runs/start", {
        method: "POST",
        body: JSON.stringify({ map, policy: "manual", record_frames: recordFrames })
      });
      state.manualRunId = started.run_id;
      state.manualPaused = false;
      state.manualKeys.clear();
      setManualControls(true);
      updateManualActionStatus();
      setLiveRunStatus("수동 주행 중");
      setStatus("방향키로 주행하세요. 종료하면 run.json을 저장합니다.", false);
      scheduleManualAction();
      return started;
    }
    if (!['agent', 'baseline'].includes(policy)) throw new Error("지원하지 않는 자동 주행 정책입니다.");
    state.automaticRunActive = true;
    setManualControls(false);
    setLiveRunStatus(`${policy === "agent" ? "Agent" : "기본 정책"} 실행 중…`);
    try {
      const path = policy === "agent" ? "/api/runs/agent" : "/api/runs/auto";
      const result = await apiRequest(path, {
        method: "POST",
        body: JSON.stringify({ map, policy, record_frames: recordFrames }),
        timeoutMs: 300000
      });
      addRunPayload(result, policy === "agent" ? "Agent 실행" : "기본 정책 실행");
      setLiveRunStatus(`${policy === "agent" ? "Agent" : "기본 정책"} 로그 저장 완료`);
      return result;
    } catch (error) {
      setLiveRunStatus(`실행 오류: ${error.message}`, true);
      setStatus(`자동 주행에 실패했습니다: ${error.message}`, true);
      throw error;
    } finally {
      state.automaticRunActive = false;
      setManualControls(Boolean(state.manualRunId));
    }
  }

  async function finishRun(reason = "manual") {
    if (!state.manualRunId) return null;
    stopManualTimer();
    state.manualPaused = true;
    setManualControls(true);
    const runId = state.manualRunId;
    try {
      const result = await apiRequest(`/api/runs/${encodeURIComponent(runId)}/finish`, {
        method: "POST",
        body: JSON.stringify({ reason }),
        timeoutMs: 30000
      });
      state.manualRunId = null;
      state.manualKeys.clear();
      setManualControls(false);
      addRunPayload(result, "수동 실행");
      setLiveRunStatus("run.json 저장 완료");
      return result;
    } catch (error) {
      setLiveRunStatus(`저장 오류: ${error.message}`, true);
      setStatus(`실행 로그를 저장하지 못했습니다: ${error.message}`, true);
      throw error;
    }
  }

  function pauseManualRun(forcePause = false) {
    if (!state.manualRunId || state.automaticRunActive) return;
    if (forcePause || !state.manualPaused) {
      state.manualPaused = true;
      stopManualTimer();
      setLiveRunStatus("일시정지");
    } else {
      state.manualPaused = false;
      setLiveRunStatus("수동 주행 중");
      scheduleManualAction();
    }
    setManualControls(true);
  }

  function stepManualRun() {
    if (!state.manualRunId || state.automaticRunActive) return;
    state.manualPaused = true;
    stopManualTimer();
    setManualControls(true);
    sendManualAction(currentManualAction()).catch(() => {});
  }

  function handleManualKey(event, pressed) {
    if (!MANUAL_KEYS.has(event.code) || !state.manualRunId) return;
    if (event.target && ["INPUT", "SELECT", "TEXTAREA"].includes(event.target.tagName)) return;
    event.preventDefault();
    if (pressed) state.manualKeys.add(event.code);
    else state.manualKeys.delete(event.code);
    updateManualActionStatus();
  }

  function changeMapKind() {
    const kind = $("map-kind").value;
    updateMapKindVisibility();
    if (kind === "custom") {
      if (state.mapSpec.map_kind !== "custom") {
        state.preview = null;
        $("map-status").textContent = "자체 맵 생성 대기";
        drawTrack($("map-canvas"), null, []);
        renderControlPointTable();
        setMapValidation("템플릿과 디자인 seed를 정한 뒤 새 트랙을 생성하세요.", false);
      } else {
        renderMap();
      }
      return;
    }
    if (state.mapSpec.map_kind === "custom") {
      state.mapSpec = normalizeMap({
        ...DEFAULT_MAP,
        map_id: `official-track-${Number($("track-id").value)}-seed-${Number($("seed").value)}`,
        track_id: $("track-id").value,
        seed: $("seed").value,
        obstacle_mode: $("obstacle-mode").value,
        obstacles: state.mapSpec.obstacles,
        max_steps: $("max-steps").value,
        frame_skip: $("frame-skip").value
      });
      state.preview = null;
    } else {
      readForm();
    }
    syncForm();
    renderMap();
  }

  function bindEvents() {
    $("map-file").addEventListener("change", (event) => {
      const file = event.target.files[0];
      if (file) readJsonFile(file, (payload) => applyMap(payload, file.name));
    });
    $("log-file").addEventListener("change", (event) => {
      const file = event.target.files[0];
      if (file) readJsonFile(file, (payload) => addRunPayload(payload, file.name));
    });
    $("add-obstacle").addEventListener("click", () => {
      try { addObstacleFromForm(); } catch (error) { setStatus(error.message, true); }
    });
    $("random-obstacles").addEventListener("click", () => {
      try { generateRandomObstacles(); } catch (error) { setStatus(error.message, true); }
    });
    $("clear-obstacles").addEventListener("click", () => { state.mapSpec.obstacles = []; renderMap(); setStatus("사용자 장애물을 모두 삭제했습니다.", false); });
    $("download-map").addEventListener("click", () => { saveMap().catch((error) => setStatus(error.message, true)); });
    $("download-map-file").addEventListener("click", () => { try { downloadMap(); } catch (error) { setStatus(error.message, true); } });
    $("map-kind").addEventListener("change", changeMapKind);
    $("generate-custom-map").addEventListener("click", () => { generateCustomMap().catch((error) => setStatus(error.message, true)); });
    $("map-form").addEventListener("change", (event) => {
      if (event.target.id === "map-kind") return;
      if ($("map-kind").value === "custom" && state.mapSpec.map_kind !== "custom") return;
      try {
        readForm();
        if (state.mapSpec.map_kind === "custom") state.preview = previewFromCustomMap(state.mapSpec);
        renderMap();
      } catch (error) { setStatus(error.message, true); }
    });
    $("custom-width").addEventListener("input", (event) => {
      if (state.mapSpec.map_kind !== "custom") return;
      try {
        state.mapSpec.geometry.width = finiteNumber(event.target.value, "도로 반폭", 0.5, 100);
        refreshCustomMapPreview();
      } catch (error) { setMapValidation(error.message, true); }
    });
    $("custom-map-id").addEventListener("input", (event) => {
      if (state.mapSpec.map_kind !== "custom") return;
      state.mapSpec.map_id = event.target.value.trim();
      refreshCustomMapPreview();
    });
    $("control-point-table").addEventListener("input", editControlPoint);
    $("add-control-point").addEventListener("click", () => { try { addControlPoint(); } catch (error) { setMapValidation(error.message, true); } });
    $("remove-control-point").addEventListener("click", () => { try { removeControlPoint(); } catch (error) { setMapValidation(error.message, true); } });
    $("start-agent-run").addEventListener("click", () => startRun($("run-policy").value).catch((error) => setStatus(error.message, true)));
    $("start-manual-run").addEventListener("click", () => startRun("manual").catch((error) => setStatus(error.message, true)));
    $("pause-manual-run").addEventListener("click", () => pauseManualRun());
    $("step-manual-run").addEventListener("click", stepManualRun);
    $("finish-manual-run").addEventListener("click", () => finishRun().catch(() => {}));
    document.addEventListener("keydown", (event) => handleManualKey(event, true));
    document.addEventListener("keyup", (event) => handleManualKey(event, false));
    document.addEventListener("visibilitychange", () => { if (document.hidden) pauseManualRun(true); });
    window.addEventListener("blur", () => { state.manualKeys.clear(); updateManualActionStatus(); });
    window.addEventListener("pagehide", () => {
      stopManualTimer();
      state.requestControllers.forEach((controller) => controller.abort());
    });
    $("obstacle-table").addEventListener("input", (event) => {
      const input = event.target.closest("[data-obstacle]");
      if (!input) return;
      try {
        const index = Number(input.dataset.obstacle);
        const key = input.dataset.key;
        const limits = { progress: [0, 1], lateral: [-1, 1], radius: [.2, 4] }[key];
        state.mapSpec.obstacles[index][key] = finiteNumber(input.value, key, limits[0], limits[1]);
        if (state.mapSpec.map_kind === "custom") state.preview = previewFromCustomMap(state.mapSpec);
        drawTrack($("map-canvas"), state.preview, state.mapSpec.obstacles);
      } catch (error) { setStatus(error.message, true); }
    });
    $("obstacle-table").addEventListener("click", (event) => {
      const button = event.target.closest(".delete-obstacle");
      if (!button) return;
      state.mapSpec.obstacles.splice(Number(button.dataset.index), 1);
      renderMap();
      setStatus("사용자 장애물을 삭제했습니다.", false);
    });
    const drop = $("file-drop");
    ["dragenter", "dragover"].forEach((name) => drop.addEventListener(name, (event) => { event.preventDefault(); drop.classList.add("dragover"); }));
    ["dragleave", "drop"].forEach((name) => drop.addEventListener(name, (event) => { event.preventDefault(); drop.classList.remove("dragover"); }));
    drop.addEventListener("drop", (event) => {
      const file = event.dataTransfer.files[0];
      if (!file) return;
      readJsonFile(file, (payload) => {
        if (Array.isArray(payload.steps) && payload.track) addRunPayload(payload, file.name);
        else applyMap(payload, file.name);
      });
    });
    $("play-button").addEventListener("click", () => setPlaybackRunning(true));
    $("pause-button").addEventListener("click", () => setPlaybackRunning(false));
    $("step-forward").addEventListener("click", () => { setPlaybackRunning(false); advancePlayback(); });
    $("step-back").addEventListener("click", () => {
      setPlaybackRunning(false);
      if (state.runState) renderRunFrame(state.runState, state.runState.stepIndex - 1);
    });
    $("restart-button").addEventListener("click", () => {
      setPlaybackRunning(false);
      if (state.runState) renderRunFrame(state.runState, 0);
    });
    $("speed").addEventListener("change", (event) => {
      if (state.runState) state.runState.speed = Number(event.target.value) || 1;
    });
    $("timeline").addEventListener("input", (event) => {
      setPlaybackRunning(false);
      if (state.runState) renderRunFrame(state.runState, Number(event.target.value));
    });
  }

  function initialize() {
    bindEvents();
    setRunControls(false);
    setManualControls(false);
    syncForm();
    renderMap();
    renderComparison();
    checkLocalApi().then((available) => {
      if (!available) setStatus("파일 보기 모드입니다. 로그 재생과 맵 JSON 저장은 계속 사용할 수 있습니다.", false);
    });
  }

  window.HAICSimulator = {
    loadRunLog,
    normalizeMap,
    renderRunFrame,
    setPlaybackRunning,
    advancePlayback,
    summarizeRuns,
    generateCustomMap,
    applyCustomMap,
    startRun,
    sendManualAction,
    finishRun,
    makeBrowserCustomMap,
    manualActionFromKeys
  };
  document.addEventListener("DOMContentLoaded", initialize);
}());

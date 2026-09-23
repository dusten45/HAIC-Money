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
    agents: [],
    selectedAgentId: "repository",
    manualRunId: null,
    manualVehicle: null,
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
    updateAgentPicker();
  }

  function updateAgentPicker() {
    const picker = $("agent-select");
    const status = $("agent-status");
    const policy = $("run-policy");
    if (!picker || !status || !policy) return;
    const agentPolicy = policy.value === "agent";
    picker.disabled = !state.apiAvailable || !agentPolicy || state.automaticRunActive || Boolean(state.manualRunId);
    status.textContent = !state.apiAvailable
      ? "로컬 실행기 연결 대기 중"
      : state.agents.length
        ? `${state.agents.length}개 Agent 사용 가능`
        : "선택 가능한 Agent가 없습니다";
    status.classList.toggle("error", state.apiAvailable && !state.agents.length);
  }

  async function loadAgents() {
    const picker = $("agent-select");
    if (!picker || !state.apiAvailable) return;
    const response = await apiRequest("/api/agents", { timeoutMs: 5000 });
    state.agents = Array.isArray(response.agents) ? response.agents : [];
    picker.replaceChildren();
    for (const agent of state.agents) {
      const option = document.createElement("option");
      option.value = String(agent.id);
      option.textContent = `${agent.name}${agent.ready ? "" : " · 모델 없음"}`;
      option.disabled = !agent.ready;
      picker.append(option);
    }
    if (!state.agents.some((agent) => agent.id === state.selectedAgentId && agent.ready)) {
      const firstReady = state.agents.find((agent) => agent.ready);
      state.selectedAgentId = firstReady ? firstReady.id : (state.agents[0]?.id || "repository");
    }
    picker.value = state.selectedAgentId;
    updateAgentPicker();
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
      if (state.apiAvailable) await loadAgents();
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

  function integerNumber(value, name, minimum, maximum) {
    if ((typeof value === "string" && value.trim() === "") ||
        (typeof value !== "string" && typeof value !== "number")) {
      throw new Error(`${name}은 정수여야 합니다.`);
    }
    const number = finiteNumber(value, name, minimum, maximum);
    if (!Number.isInteger(number)) throw new Error(`${name}은 정수여야 합니다.`);
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
      if (centerline.length > 4096) throw new Error("중심선은 최대 4096개 점까지 지원합니다.");
      const direction = Number(rawGeometry.direction ?? 1);
      if (![1, -1].includes(direction)) throw new Error("주행 방향은 1 또는 -1이어야 합니다.");
      const geometryWidth = finiteNumber(rawGeometry.width, "width", 0.5, 100);
      const oversizedObstacle = obstacles.find((obstacle) => obstacle.radius > geometryWidth);
      if (oversizedObstacle) {
        throw new Error(`obstacle radius ${oversizedObstacle.radius} exceeds custom road width ${geometryWidth}`);
      }
      return {
        ...common,
        map_kind: "custom",
        map_id: mapId,
        geometry: {
          centerline,
          width: geometryWidth,
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
    const pointCount = centerline.length;
    const startIndex = Number(map.geometry.start_index ?? 0);
    const direction = Number(map.geometry.direction ?? 1);
    const orderedCenterline = Array.from({ length: pointCount }, (_unused, index) =>
      centerline[(startIndex + direction * index + pointCount) % pointCount]
    );
    const points = orderedCenterline.map(([x, y], index) => {
      const previous = orderedCenterline[(index - 1 + pointCount) % pointCount];
      const following = orderedCenterline[(index + 1) % pointCount];
      const tangent = Math.atan2(following[1] - previous[1], following[0] - previous[0]);
      return [index / pointCount, tangent - Math.PI / 2, x, y];
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
    if (!Number.isFinite(geometry.width) || geometry.width < 0.5 || geometry.width > 100) return "도로 반폭은 0.5에서 100 사이여야 합니다.";
    if (!/^custom-track-[A-Za-z0-9][A-Za-z0-9_-]{0,63}$/.test(map.map_id || "")) return "맵 ID는 custom-track-으로 시작해야 합니다.";
    const oversizedObstacle = (Array.isArray(map.obstacles) ? map.obstacles : [])
      .find((obstacle) => obstacle && obstacle.radius > geometry.width);
    if (oversizedObstacle) return `장애물 반경 ${oversizedObstacle.radius}이 도로 반폭 ${geometry.width}보다 큽니다.`;
    try {
      validateBrowserCustomGeometry(points, geometry.width);
      return "";
    } catch (error) {
      return error.message;
    }
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
    if (points.length > 256) {
      table.innerHTML = '<tr><td colspan="3" class="empty-cell">중심선 샘플이 많아 직접 편집 표는 숨겼습니다. 트랙 형태를 바꾸려면 템플릿이나 seed를 조정하세요.</td></tr>';
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
    state.mapSpec.generator = {};
    updateGeneratedTrackSummary(state.mapSpec);
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
    state.mapSpec.generator = {};
    updateGeneratedTrackSummary(state.mapSpec);
    renderMap();
  }

  function removeControlPoint() {
    if (state.mapSpec.map_kind !== "custom") return;
    const points = state.mapSpec.geometry.centerline;
    if (points.length <= 12) throw new Error("유효성 검사를 위해 중심선 점을 최소 12개 유지해야 합니다.");
    points.pop();
    state.mapSpec.generator = {};
    updateGeneratedTrackSummary(state.mapSpec);
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

  function drawTrack(canvas, preview, obstacles, emptyId = "canvas-empty", vehicle = null) {
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
      const maxOffset = Math.max(0, Math.min(preview.track.width * .6, preview.track.width - obstacle.radius));
      const offset = obstacle.lateral * maxOffset;
      drawObstacle([trackPoint[2] + offset * Math.cos(beta), trackPoint[3] + offset * Math.sin(beta)], obstacle.radius, "#7ce2b2");
    });
    const start = mapped[0];
    context.beginPath();
    context.arc(start.x, start.y, 5, 0, Math.PI * 2);
    context.fillStyle = "#8eb5ff";
    context.fill();
    if (vehicle && Array.isArray(vehicle.position) && vehicle.position.length === 2) {
      const position = project(vehicle.position, bounds, width, height);
      context.save();
      context.translate(position.x, position.y);
      context.rotate(-Number(vehicle.angle || 0));
      context.beginPath();
      context.moveTo(11, 0);
      context.lineTo(-7, 6);
      context.lineTo(-5, 0);
      context.lineTo(-7, -6);
      context.closePath();
      context.fillStyle = "#ff6f74";
      context.fill();
      context.strokeStyle = "#fff";
      context.lineWidth = 1.5;
      context.stroke();
      context.restore();
    }
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
    drawTrack($("map-canvas"), state.preview, state.mapSpec.obstacles, "canvas-empty", state.manualVehicle);
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

  function formatAction(action) {
    if (!Array.isArray(action) || action.length !== 3 || action.some((value) => !Number.isFinite(Number(value)))) {
      return "—";
    }
    return action.map((value) => Number(value).toFixed(2)).join(" / ");
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
    $("metric-action").textContent = formatAction(current && current.action);
    const speed = current && Array.isArray(current.velocity)
      ? Math.hypot(Number(current.velocity[0]), Number(current.velocity[1]))
      : NaN;
    $("metric-speed").textContent = Number.isFinite(speed) ? `${speed.toFixed(2)} m/s` : "—";
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

  function renderComparison(logs = state.logs) {
    const rows = summarizeRuns(logs);
    const table = $("run-table");
    if (!rows.length) {
      const row = document.createElement("tr");
      const cell = document.createElement("td");
      cell.colSpan = 6;
      cell.className = "empty-cell";
      cell.textContent = "로그를 추가하면 기록이 쌓입니다.";
      row.append(cell);
      table.replaceChildren(row);
      return;
    }
    const tableRows = rows.map((row) => {
      const tableRow = document.createElement("tr");
      const runIdCell = document.createElement("td");
      runIdCell.title = row.run_id;
      runIdCell.textContent = row.run_id.slice(0, 10);
      tableRow.append(runIdCell);

      const resultCell = document.createElement("td");
      resultCell.className = row.finished ? "finished" : "dnf";
      resultCell.textContent = row.finished ? "FINISHED" : "DNF";
      tableRow.append(resultCell);

      for (const value of [
        formatLapTime(row.lap_time_ms),
        formatPercent(row.progress),
        formatPercent(row.damage),
        String(row.collision_count)
      ]) {
        const cell = document.createElement("td");
        cell.textContent = value;
        tableRow.append(cell);
      }
      return tableRow;
    });
    table.replaceChildren(...tableRows);
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
    updateGeneratedTrackSummary(state.mapSpec);
    setStatus(`${source}을 불러왔습니다.`, false);
  }

  function updateGeneratedTrackSummary(map) {
    const target = $("generated-track-summary");
    if (!target) return;
    target.textContent = formatGeneratedTrackSummary(map);
  }

  function formatGeneratedTrackSummary(map) {
    if (!map || map.map_kind !== "custom") {
      return "공식 트랙";
    }
    const generator = map.generator || {};
    if (generator.template === "extreme_technical" &&
        Number.isInteger(generator.corner_count) &&
        Number.isInteger(generator.s_section_count) &&
        Number.isInteger(generator.near_90_corner_count)) {
      return `${generator.corner_count}개 코너 · S자 ${generator.s_section_count}구간 · 90° 급코너 ${generator.near_90_corner_count}개`;
    }
    const sequence = generator.corner_sequence;
    if (!Number.isInteger(generator.corner_count) || !Array.isArray(sequence) ||
        sequence.length !== generator.corner_count ||
        !sequence.every((token) => typeof token === "string" && /^(left|right):(wide|medium|tight|hairpin)$/.test(token))) {
      return "직접 제작 맵";
    }
    const directions = sequence.map((token) => token.split(":", 1)[0]);
    const switches = directions.reduce((count, direction, index) =>
      count + (index > 0 && direction !== directions[index - 1] ? 1 : 0), 0);
    const hairpins = sequence.filter((token) => token.endsWith(":hairpin")).length;
    return `${generator.corner_count}개 코너 · 헤어핀 ${hairpins} · 방향 전환 ${switches}회`;
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

  const TRACK_TEMPLATES = new Set(["oval", "s_curve", "hairpin", "chicane", "technical", "extreme_technical"]);
  const CORNER_COUNT_RANGES = {
    oval: [4, 4], s_curve: [6, 8], hairpin: [6, 9], chicane: [7, 10], technical: [9, 12],
    extreme_technical: [12, 16]
  };
  const MAX_CENTERLINE_POINTS = 4096;
  const MAX_GENERATED_TRACK_WIDTH = 9;

  function createTrackRandom(seed) {
    let value = seed >>> 0;
    if (value === 0) value = 1;
    const random = () => {
      value = (Math.imul(value, 1664525) + 1013904223) >>> 0;
      return value / 4294967296;
    };
    random.index = (length) => Math.min(length - 1, Math.floor(random() * length));
    random.uniform = (minimum, maximum) => minimum + (maximum - minimum) * random();
    return random;
  }

  function cornerSequence(template, random) {
    if (!TRACK_TEMPLATES.has(template)) {
      throw new Error(`template must be one of ${Array.from(TRACK_TEMPLATES).sort().join(", ")}`);
    }
    const [minimum, maximum] = CORNER_COUNT_RANGES[template];
    if (template === "extreme_technical") random();
    const count = template === "extreme_technical"
      ? 12 + 2 * random.index(3)
      : minimum + random.index(maximum - minimum + 1);
    if (template === "oval") return Array(count).fill("left:wide");
    if (template === "s_curve") {
      const directions = ["left", "right", "left", "left", "right", "left"];
      while (directions.length < count) {
        directions.push(directions[directions.length - 1] === "left" ? "right" : "left");
      }
      const classes = ["wide", "medium", "tight"];
      return directions.map((direction) => `${direction}:${classes[random.index(classes.length)]}`);
    }
    if (template === "hairpin") {
      const sequence = Array.from({ length: count }, () => `left:${random() < 0.4 ? "wide" : "medium"}`);
      const hairpinIndices = [Math.floor(count / 3), Math.floor((2 * count) / 3)];
      const rightHairpin = hairpinIndices[random.index(2)];
      for (const index of hairpinIndices) {
        sequence[index] = `${index === rightHairpin ? "right" : "left"}:hairpin`;
      }
      return sequence;
    }
    if (template === "chicane") {
      const directions = Array.from({ length: count }, (_item, index) => index % 2 === 0 ? "left" : "right");
      if (count % 2 === 0) directions[count - 1] = "left";
      return Array.from({ length: count }, (_item, index) => {
        return `${directions[index]}:${random() < 0.65 ? "tight" : "medium"}`;
      });
    }
    if (template === "extreme_technical") {
      const hairpinCount = Math.floor(count / 4);
      const hairpinIndices = new Set(
        Array.from({ length: hairpinCount }, (_item, index) => 1 + Math.floor(index * count / hairpinCount))
      );
      return Array.from({ length: count }, (_item, index) => {
        const direction = hairpinIndices.has(index) ? "right" : "left";
        const cornerClass = direction === "right" ? "hairpin" : "medium";
        return `${direction}:${cornerClass}`;
      });
    }
    const classes = ["wide", "medium", "tight"];
    for (let index = classes.length; index < count; index += 1) {
      classes.push(classes[random.index(classes.length)]);
    }
    for (let index = classes.length - 1; index > 0; index -= 1) {
      const swapIndex = random.index(index + 1);
      [classes[index], classes[swapIndex]] = [classes[swapIndex], classes[index]];
    }
    const directions = Array.from({ length: count }, (_item, index) => index % 2 === 0 ? "left" : "right");
    if (count % 2 === 0) directions[count - 1] = "left";
    return classes.map((cornerClass, index) => `${directions[index]}:${cornerClass}`);
  }

  function trackDistance(first, second) {
    return Math.hypot(second[0] - first[0], second[1] - first[1]);
  }

  function trackInterpolate(start, end, ratio) {
    return [start[0] + (end[0] - start[0]) * ratio, start[1] + (end[1] - start[1]) * ratio];
  }

  function quadraticBezier(start, control, end, ratio) {
    const inverse = 1 - ratio;
    return [
      inverse * inverse * start[0] + 2 * inverse * ratio * control[0] + ratio * ratio * end[0],
      inverse * inverse * start[1] + 2 * inverse * ratio * control[1] + ratio * ratio * end[1]
    ];
  }

  function angleGaps(random, count) {
    const weights = Array.from({ length: count }, () => 0.5 + random());
    const weightTotal = weights.reduce((sum, weight) => sum + weight, 0);
    const rawGaps = weights.map((weight) => 2 * Math.PI * weight / weightTotal);
    const minimum = 20 * Math.PI / 180;
    const maximum = 100 * Math.PI / 180;
    const base = 2 * Math.PI / count;
    let factor = 1;
    for (const gap of rawGaps) {
      const deviation = gap - base;
      if (deviation > 0) factor = Math.min(factor, (maximum - base) / deviation);
      else if (deviation < 0) factor = Math.min(factor, (base - minimum) / -deviation);
    }
    factor = Math.max(0, Math.min(1, factor * 0.999));
    return rawGaps.map((gap) => base + factor * (gap - base));
  }

  function profileAngleGaps(gaps, sequence, compactSTurns = false) {
    const constrainedCorners = [];
    sequence.forEach((token, index) => {
      if (token.startsWith("right:") || token.endsWith(":hairpin")) constrainedCorners.push(index);
    });
    if (!constrainedCorners.length) return gaps;

    const count = sequence.length;
    const constrainedGaps = new Set();
    for (const index of constrainedCorners) {
      const adjacent = [(index - 1 + count) % count, index];
      if (adjacent.some((gapIndex) => constrainedGaps.has(gapIndex))) {
        throw new Error("corner profile contains adjacent concave turns");
      }
      adjacent.forEach((gapIndex) => constrainedGaps.add(gapIndex));
    }
    const remainingIndices = Array.from({ length: count }, (_item, index) => index)
      .filter((index) => !constrainedGaps.has(index));
    if (!remainingIndices.length) throw new Error("corner profile has no unconstrained straight intervals");
    const minimumPairTotal = Math.max(40, (360 - 100 * remainingIndices.length) / constrainedCorners.length);
    const maximumPairTotal = Math.min(200, (360 - 20 * remainingIndices.length) / constrainedCorners.length);
    if (maximumPairTotal < minimumPairTotal) throw new Error("corner profile cannot fit within the angle-gap bounds");
    const preferredPairTotal = compactSTurns ? 60 : 80;
    const pairTotal = Math.max(minimumPairTotal, Math.min(preferredPairTotal, maximumPairTotal));
    const remainingTotal = 360 - pairTotal * constrainedCorners.length;
    const remainingBase = remainingTotal / remainingIndices.length;
    if (remainingBase < 20 || remainingBase > 100) throw new Error("corner profile cannot fit within the angle-gap bounds");

    const rawDegrees = remainingIndices.map((index) => gaps[index] * 180 / Math.PI);
    const rawMean = rawDegrees.reduce((sum, gap) => sum + gap, 0) / rawDegrees.length;
    let factor = 1;
    for (const gap of rawDegrees) {
      const deviation = gap - rawMean;
      if (deviation > 0) factor = Math.min(factor, (100 - remainingBase) / deviation);
      else if (deviation < 0) factor = Math.min(factor, (remainingBase - 20) / -deviation);
    }
    factor = Math.max(0, Math.min(1, factor * 0.999));
    const adjusted = Array(count).fill(pairTotal * 0.5 * Math.PI / 180);
    remainingIndices.forEach((index, position) => {
      adjusted[index] = (remainingBase + factor * (rawDegrees[position] - rawMean)) * Math.PI / 180;
    });
    return adjusted;
  }

  function roundTrackMetric(value, places) {
    const scale = 10 ** places;
    const magnitude = Math.floor(Math.abs(value) * scale + 0.5) / scale;
    return value < 0 ? -magnitude : magnitude;
  }

  function measureCornerProfiles(centerline, ranges, width) {
    const turns = [];
    const radii = [];
    const count = centerline.length;
    for (const [start, end] of ranges) {
      let totalTurn = 0;
      const localRadii = [];
      for (let index = start; index <= end; index += 1) {
        const previous = centerline[(index - 1 + count) % count];
        const point = centerline[index % count];
        const following = centerline[(index + 1) % count];
        const incoming = [point[0] - previous[0], point[1] - previous[1]];
        const outgoing = [following[0] - point[0], following[1] - point[1]];
        const cross = incoming[0] * outgoing[1] - incoming[1] * outgoing[0];
        const dot = incoming[0] * outgoing[0] + incoming[1] * outgoing[1];
        const turn = Math.atan2(cross, dot);
        totalTurn += turn;
        const sideA = Math.hypot(...incoming);
        const sideB = Math.hypot(...outgoing);
        const sideC = trackDistance(previous, following);
        if (start < index && index < end && Math.abs(cross) > 1e-9 && Math.abs(turn) > Math.PI / 1800) {
          localRadii.push(sideA * sideB * sideC / (2 * Math.abs(cross)));
        }
      }
      turns.push(roundTrackMetric(totalTurn * 180 / Math.PI, 2));
      radii.push(roundTrackMetric((localRadii.length ? Math.min(...localRadii) : 1e6) / width, 3));
    }
    return { turns, radii };
  }

  function roundTrackCoordinate(value) {
    const magnitude = Math.floor(Math.abs(value) * 100000 + 0.5) / 100000;
    const rounded = value < 0 ? -magnitude : magnitude;
    return rounded === 0 ? 0 : rounded;
  }

  function segmentCross(first, second, third) {
    return (second[0] - first[0]) * (third[1] - first[1]) -
      (second[1] - first[1]) * (third[0] - first[0]);
  }

  function pointSegmentDistance(point, start, end) {
    const dx = end[0] - start[0];
    const dy = end[1] - start[1];
    const lengthSquared = dx * dx + dy * dy;
    if (lengthSquared <= 1e-18) return trackDistance(point, start);
    const ratio = Math.max(0, Math.min(1,
      ((point[0] - start[0]) * dx + (point[1] - start[1]) * dy) / lengthSquared));
    return trackDistance(point, [start[0] + ratio * dx, start[1] + ratio * dy]);
  }

  function segmentsDistance(firstStart, firstEnd, secondStart, secondEnd) {
    if (segmentsIntersect(firstStart, firstEnd, secondStart, secondEnd)) return 0;
    return Math.min(
      pointSegmentDistance(firstStart, secondStart, secondEnd),
      pointSegmentDistance(firstEnd, secondStart, secondEnd),
      pointSegmentDistance(secondStart, firstStart, firstEnd),
      pointSegmentDistance(secondEnd, firstStart, firstEnd)
    );
  }

  function areAdjacent(firstIndex, secondIndex, count) {
    return firstIndex === secondIndex ||
      (firstIndex + 1) % count === secondIndex ||
      (secondIndex + 1) % count === firstIndex;
  }

  function roadBoundaries(centerline, width) {
    const left = [];
    const right = [];
    for (let index = 0; index < centerline.length; index += 1) {
      const point = centerline[index];
      const previous = centerline[(index - 1 + centerline.length) % centerline.length];
      const following = centerline[(index + 1) % centerline.length];
      const incomingLength = trackDistance(previous, point);
      const outgoingLength = trackDistance(point, following);
      const incoming = [(point[0] - previous[0]) / incomingLength, (point[1] - previous[1]) / incomingLength];
      const outgoing = [(following[0] - point[0]) / outgoingLength, (following[1] - point[1]) / outgoingLength];
      const incomingNormal = [-incoming[1], incoming[0]];
      const outgoingNormal = [-outgoing[1], outgoing[0]];
      let normalX = incomingNormal[0] + outgoingNormal[0];
      let normalY = incomingNormal[1] + outgoingNormal[1];
      let normalLength = Math.hypot(normalX, normalY);
      if (normalLength <= 1e-9) {
        normalX = outgoingNormal[0];
        normalY = outgoingNormal[1];
        normalLength = 1;
      }
      const normal = [normalX / normalLength, normalY / normalLength];
      const denominator = Math.abs(normal[0] * incomingNormal[0] + normal[1] * incomingNormal[1]);
      const miterLength = Math.min(width / Math.max(denominator, 1e-6), width * 4);
      const offset = [normal[0] * miterLength, normal[1] * miterLength];
      left.push([point[0] + offset[0], point[1] + offset[1]]);
      right.push([point[0] - offset[0], point[1] - offset[1]]);
    }
    return [left, right];
  }

  function roadEdgesIntersect(centerline, width) {
    const count = centerline.length;
    for (const boundary of roadBoundaries(centerline, width)) {
      for (let firstIndex = 0; firstIndex < count; firstIndex += 1) {
        const firstStart = boundary[firstIndex];
        const firstEnd = boundary[(firstIndex + 1) % count];
        for (let secondIndex = firstIndex + 1; secondIndex < count; secondIndex += 1) {
          if (areAdjacent(firstIndex, secondIndex, count)) continue;
          const secondStart = boundary[secondIndex];
          const secondEnd = boundary[(secondIndex + 1) % count];
          if (
            Math.max(firstStart[0], firstEnd[0]) + 1e-9 < Math.min(secondStart[0], secondEnd[0]) ||
            Math.max(secondStart[0], secondEnd[0]) + 1e-9 < Math.min(firstStart[0], firstEnd[0]) ||
            Math.max(firstStart[1], firstEnd[1]) + 1e-9 < Math.min(secondStart[1], secondEnd[1]) ||
            Math.max(secondStart[1], secondEnd[1]) + 1e-9 < Math.min(firstStart[1], firstEnd[1])
          ) continue;
          if (segmentsIntersect(firstStart, firstEnd, secondStart, secondEnd)) return true;
        }
      }
    }

    const lengths = centerline.map((point, index) => trackDistance(point, centerline[(index + 1) % count]));
    const totalLength = lengths.reduce((sum, length) => sum + length, 0);
    const midpointDistances = [];
    let cumulative = 0;
    for (const length of lengths) {
      midpointDistances.push(cumulative + length * 0.5);
      cumulative += length;
    }
    const requiredArcSeparation = Math.max(width * 4, 8);
    const minimumCenterlineDistance = width * 2;
    for (let firstIndex = 0; firstIndex < count; firstIndex += 1) {
      const firstMidpoint = midpointDistances[firstIndex];
      const firstStart = centerline[firstIndex];
      const firstEnd = centerline[(firstIndex + 1) % count];
      for (let secondIndex = firstIndex + 1; secondIndex < count; secondIndex += 1) {
        if (areAdjacent(firstIndex, secondIndex, count)) continue;
        let separation = Math.abs(midpointDistances[secondIndex] - firstMidpoint);
        separation = Math.min(separation, totalLength - separation);
        if (separation <= requiredArcSeparation) continue;
        const secondStart = centerline[secondIndex];
        const secondEnd = centerline[(secondIndex + 1) % count];
        if (segmentsDistance(firstStart, firstEnd, secondStart, secondEnd) < minimumCenterlineDistance) return true;
      }
    }
    return false;
  }

  function validateBrowserCustomGeometry(centerline, width) {
    if (!Array.isArray(centerline) || centerline.length < 12) throw new Error("centerline must contain at least 12 points");
    if (centerline.length > MAX_CENTERLINE_POINTS) throw new Error(`centerline must contain at most ${MAX_CENTERLINE_POINTS} points`);
    if (!Number.isFinite(width) || width <= 0) throw new Error("width must be positive and finite");
    const lengths = centerline.map((point, index) => trackDistance(point, centerline[(index + 1) % centerline.length]));
    if (lengths.some((length) => length <= 1e-6)) throw new Error("centerline contains a zero-length segment");
    if (lengths.reduce((sum, length) => sum + length, 0) < width * 8) throw new Error("centerline is too short for the selected width");
    for (let firstIndex = 0; firstIndex < centerline.length; firstIndex += 1) {
      for (let secondIndex = firstIndex + 1; secondIndex < centerline.length; secondIndex += 1) {
        if (areAdjacent(firstIndex, secondIndex, centerline.length)) continue;
        if (segmentsIntersect(
          centerline[firstIndex], centerline[(firstIndex + 1) % centerline.length],
          centerline[secondIndex], centerline[(secondIndex + 1) % centerline.length]
        )) throw new Error(`centerline self-intersection between segments ${firstIndex} and ${secondIndex}`);
      }
    }
    for (let index = 0; index < centerline.length; index += 1) {
      const point = centerline[index];
      const previous = centerline[(index - 1 + centerline.length) % centerline.length];
      const following = centerline[(index + 1) % centerline.length];
      const incoming = [point[0] - previous[0], point[1] - previous[1]];
      const outgoing = [following[0] - point[0], following[1] - point[1]];
      const cosine = (incoming[0] * outgoing[0] + incoming[1] * outgoing[1]) /
        (lengths[(index - 1 + lengths.length) % lengths.length] * lengths[index]);
      if (cosine < -0.995) throw new Error(`centerline turn is too sharp at point ${index}`);
      const doubledArea = Math.abs(segmentCross(previous, point, following));
      if (doubledArea > 1e-9) {
        const radius = lengths[(index - 1 + lengths.length) % lengths.length] * lengths[index] *
          trackDistance(previous, following) / (2 * doubledArea);
        const minimumRadius = width * 1.05;
        if (radius < minimumRadius) {
          throw new Error(`turn radius ${radius} is below minimum ${minimumRadius} at point ${index}`);
        }
      }
    }
    if (roadEdgesIntersect(centerline, width)) throw new Error("road boundaries intersect or overlap");
  }

  const CORNER_RADIUS_WIDTH_RANGES = {
    wide: [1.45, 3.2], medium: [1.1, 2.7], tight: [1.45, 2.0], hairpin: [1.45, 1.9]
  };

  function buildBrowserGeometryCandidate(template, designSeed, width) {
    const random = createTrackRandom(designSeed);
    const sequence = cornerSequence(template, random);
    const count = sequence.length;
    const gaps = profileAngleGaps(
      angleGaps(random, count),
      sequence,
      template === "extreme_technical"
    );
    const radiusX = 150 + (width - 8) + random.uniform(-6, 6);
    let radiusY = 93.75 + (width - 8) * 0.625 + random.uniform(-4, 4);
    const roundness = Math.max(0, Math.min(1, (width - 8) / 92));
    const targetAxisRatio = template !== "oval" ? 1 : 1.6 - 0.45 * roundness;
    radiusY = Math.max(radiusY, radiusX / targetAxisRatio);
    const phase = random.uniform(0, 2 * Math.PI);
    const templateAmplitude = {
      oval: 0.015, s_curve: 0.02, hairpin: 0.02, chicane: 0.03, technical: 0.04, extreme_technical: 0.055
    }[template];
    const classAdjustment = { wide: 0.01, medium: 0, tight: -0.01, hairpin: 0 };
    const directions = sequence.map((token) => token.split(":", 1)[0]);
    const wideTrackRatio = Math.max(0, Math.min(1, (width - 8) / 92));
    const regularLeftRadius = 1.15 - 0.1 * wideTrackRatio;
    const regularRightRadius = 0.5 + 0.25 * wideTrackRatio;
    const nestedLeftRadius = 0.9 + 0.15 * wideTrackRatio;
    const anchors = [];
    let angle = random.uniform(0, 2 * Math.PI);
    for (let index = 0; index < count; index += 1) {
      const [direction, cornerClass] = sequence[index].split(":", 2);
      const directionSign = direction === "left" ? 1 : -1;
      const radialModulation = templateAmplitude * Math.sin((1 + (index % 3)) * angle + phase);
      const radialJitter = random.uniform(-0.025, 0.025);
      let directionRadius;
      if (cornerClass === "hairpin") {
        directionRadius = direction === "left" ? 1.8 - 0.5 * wideTrackRatio : 0.04 + 0.7 * wideTrackRatio;
        anchors.push([
          radiusX * (directionRadius + radialJitter) * Math.cos(angle),
          radiusY * (directionRadius + radialJitter) * Math.sin(angle)
        ]);
      } else {
        directionRadius = direction === "left" && directions[(index - 1 + count) % count] === "right" &&
          directions[(index + 1) % count] === "right"
          ? nestedLeftRadius
          : (direction === "left" ? regularLeftRadius : regularRightRadius);
        const radialScale = directionRadius + directionSign * classAdjustment[cornerClass] + radialModulation + radialJitter;
        anchors.push([radiusX * radialScale * Math.cos(angle), radiusY * radialScale * Math.sin(angle)]);
      }
      angle += gaps[index];
    }

    const incomingPoints = [];
    const outgoingPoints = [];
    for (let index = 0; index < count; index += 1) {
      const anchor = anchors[index];
      const previous = anchors[(index - 1 + count) % count];
      const following = anchors[(index + 1) % count];
      const incomingLength = trackDistance(anchor, previous);
      const outgoingLength = trackDistance(anchor, following);
      const previousRay = [(previous[0] - anchor[0]) / incomingLength, (previous[1] - anchor[1]) / incomingLength];
      const followingRay = [(following[0] - anchor[0]) / outgoingLength, (following[1] - anchor[1]) / outgoingLength];
      const interiorCosine = Math.max(-1, Math.min(1, previousRay[0] * followingRay[0] + previousRay[1] * followingRay[1]));
      const halfDeflection = (Math.PI - Math.acos(interiorCosine)) * 0.5;
      const cornerClass = sequence[index].split(":", 2)[1];
      const radiusTarget = { wide: 2.6, medium: 2.2, tight: 1.7, hairpin: 1.6 }[cornerClass] *
        width * (1 + 0.4 * Math.max(0, Math.min(1, (width - 8) / 92)));
      const quadraticRadiusFactor = Math.sin(halfDeflection) / Math.max(Math.cos(halfDeflection) ** 2, 1e-9);
      const trim = Math.min(radiusTarget * quadraticRadiusFactor, 0.45 * Math.min(incomingLength, outgoingLength));
      incomingPoints.push(trackInterpolate(anchor, previous, trim / incomingLength));
      outgoingPoints.push(trackInterpolate(anchor, following, trim / outgoingLength));
    }

    const points = [incomingPoints[0]];
    const ranges = [];
    for (let index = 0; index < count; index += 1) {
      const cornerStart = points.length - 1;
      const anchor = anchors[index];
      const outgoing = outgoingPoints[index];
      const controlLength = Math.max(trackDistance(anchor, incomingPoints[index]), trackDistance(anchor, outgoing));
      const curveSteps = Math.max(4, Math.ceil(2 * controlLength / 4));
      for (let step = 1; step <= curveSteps; step += 1) {
        points.push(quadraticBezier(incomingPoints[index], anchor, outgoing, step / curveSteps));
      }
      ranges.push([cornerStart, points.length - 1]);
      const nextIndex = (index + 1) % count;
      const straightStart = outgoingPoints[index];
      const straightEnd = incomingPoints[nextIndex];
      const straightSteps = Math.max(1, Math.ceil(trackDistance(straightStart, straightEnd) / 4));
      const lastStep = nextIndex !== 0 ? straightSteps : straightSteps - 1;
      for (let step = 1; step <= lastStep; step += 1) {
        points.push(trackInterpolate(straightStart, straightEnd, step / straightSteps));
      }
    }
    const centerline = points.map(([x, y]) => [roundTrackCoordinate(x), roundTrackCoordinate(y)]);
    if (centerline.length > MAX_CENTERLINE_POINTS) throw new Error(`generated centerline exceeds ${MAX_CENTERLINE_POINTS} points`);
    const measured = measureCornerProfiles(centerline, ranges, width);
    for (let index = 0; index < count; index += 1) {
      const token = sequence[index];
      const expectedSign = token.startsWith("left:") ? 1 : -1;
      const minimumTurn = token.endsWith(":hairpin") ? 90 : 20;
      if (expectedSign * measured.turns[index] < minimumTurn) throw new Error(`measured corner turn does not match ${token}`);
      const cornerClass = token.split(":", 2)[1];
      const [minimumRadius, maximumRadius] = CORNER_RADIUS_WIDTH_RANGES[cornerClass];
      if (measured.radii[index] < minimumRadius || measured.radii[index] > maximumRadius) {
        throw new Error(`measured corner radius does not match ${token}`);
      }
    }
    validateBrowserCustomGeometry(centerline, width);
    return { centerline, sequence, turns: measured.turns, radii: measured.radii };
  }

  function generateBrowserGeometry(template, designSeed, width) {
    if (template === "extreme_technical") return generateBrowserExtremeGeometry(designSeed, width);
    let lastError;
    for (let attempt = 0; attempt < 64; attempt += 1) {
      const attemptSeed = (designSeed + attempt * 0x9e3779b9) >>> 0;
      try {
        return buildBrowserGeometryCandidate(template, attemptSeed, width);
      } catch (error) {
        lastError = error;
      }
    }
    throw new Error(`could not generate a valid ${template} track: ${lastError.message}`);
  }

  function buildBrowserExtremeRoute(designSeed, width, attemptIndex) {
    const countRandom = createTrackRandom(designSeed);
    countRandom();
    const count = 12 + 2 * countRandom.index(3);
    const rng = createTrackRandom((designSeed + attemptIndex * 0x9E3779B9) >>> 0);
    const radiusX = 150 + (width - 8) + rng.uniform(-6, 6);
    const radiusY = 93.75 + 0.625 * (width - 8) + rng.uniform(-4, 4);
    let corners = [[-radiusX, -radiusY], [radiusX, -radiusY], [radiusX, radiusY], [-radiusX, radiusY]];
    if (rng.index(2)) corners = corners.slice().reverse().map(([x, y]) => [x, -y]);
    const shift = rng.index(4);
    corners = corners.slice(shift).concat(corners.slice(0, shift));
    const vectors = corners.map((start, index) => {
      const end = corners[(index + 1) % 4];
      const length = trackDistance(start, end);
      return [(end[0] - start[0]) / length, (end[1] - start[1]) / length];
    });
    const roundness = Math.max(0, Math.min(1, (width - 8) / 92));
    const tangent = 1.7 * width * (1 + 0.4 * roundness) * Math.sqrt(2);
    let doglegCorner = null;
    let selectedSides;
    if (count === 12) {
      selectedSides = rng.index(2) === 0 ? [0, 2] : [1, 3];
    } else if (count === 14) {
      doglegCorner = rng.index(4);
      selectedSides = [0, 1, 2, 3].filter((side) => side !== doglegCorner && side !== (doglegCorner - 1 + 4) % 4);
    } else {
      const sideLengths = corners.map((point, index) => trackDistance(point, corners[(index + 1) % 4]));
      const longest = Math.max(...sideLengths);
      const longSides = sideLengths.map((length, index) => length === longest ? index : -1).filter((index) => index >= 0);
      const shortSides = [0, 1, 2, 3].filter((side) => !longSides.includes(side));
      selectedSides = [...longSides, shortSides[rng.index(shortSides.length)]];
    }
    const sideEvents = new Map();
    const featurePoints = [];
    for (const side of selectedSides) {
      const start = corners[side], end = corners[(side + 1) % 4];
      const length = trackDistance(start, end), u = vectors[side], inward = [-u[1], u[0]];
      const sGap = rng.uniform(0.6 * width, 2.2 * width);
      const pGap = rng.uniform(0.6 * width, 1.4 * width);
      const depth = 2 * tangent + sGap, leg = 2 * tangent + pGap;
      const usable = length - 12 * width - leg;
      if (usable < 0) throw new Error("extreme notch does not fit its side");
      let position;
      if (count === 14) {
        const nearStart = side === (doglegCorner + 1) % 4;
        const jitter = rng.uniform(0, Math.min(2 * width, usable));
        position = nearStart ? 6 * width + jitter : length - 6 * width - leg - jitter;
      } else if (count === 16 && selectedSides.slice(0, 2).includes(side)) {
        const shortSide = selectedSides[2];
        const nearStart = (side + 1) % 4 === shortSide;
        const jitter = rng.uniform(0, Math.min(2 * width, usable));
        position = nearStart ? 6 * width + jitter : length - 6 * width - leg - jitter;
      } else {
        position = 6 * width + rng() * usable;
      }
      sideEvents.set(side, { position, depth, leg });
    }
    const vertices = [];
    for (let side = 0; side < 4; side += 1) {
      const start = corners[side];
      if (!vertices.length) vertices.push(start);
      if (sideEvents.has(side)) {
        const { position, depth, leg } = sideEvents.get(side);
        const u = vectors[side], inward = [-u[1], u[0]];
        const first = [start[0] + u[0] * position, start[1] + u[1] * position];
        const second = [first[0] + inward[0] * depth, first[1] + inward[1] * depth];
        const third = [second[0] + u[0] * leg, second[1] + u[1] * leg];
        const fourth = [third[0] - inward[0] * depth, third[1] - inward[1] * depth];
        vertices.push(first, second, third, fourth);
        featurePoints.push([first, second], [third, fourth]);
      }
      vertices.push(corners[(side + 1) % 4]);
    }
    if (vertices.length > 1 && vertices[vertices.length - 1][0] === vertices[0][0] && vertices[vertices.length - 1][1] === vertices[0][1]) vertices.pop();
    if (count === 14) {
      const u = vectors[(doglegCorner - 1 + 4) % 4], v = vectors[doglegCorner];
      const a = 2 * tangent + rng.uniform(0.6 * width, 2.2 * width);
      const b = 2 * tangent + rng.uniform(0.6 * width, 2.2 * width);
      const corner = corners[doglegCorner];
      const entry = [corner[0] - a * u[0], corner[1] - a * u[1]];
      const elbow = [entry[0] + b * v[0], entry[1] + b * v[1]];
      const exit = [corner[0] + b * v[0], corner[1] + b * v[1]];
      const index = vertices.findIndex(([x, y]) => x === corner[0] && y === corner[1]);
      vertices.splice(index, 1, entry, elbow, exit);
      featurePoints.push([entry, elbow]);
    }
    const indices = new Map(vertices.map((point, index) => [JSON.stringify(point), index]));
    const sSectionPairs = featurePoints.map(([first, second]) => [indices.get(JSON.stringify(first)), indices.get(JSON.stringify(second))]);
    const cornerSequence = vertices.map((point, index) => {
      const previous = vertices[(index - 1 + vertices.length) % vertices.length];
      const following = vertices[(index + 1) % vertices.length];
      const incoming = [point[0] - previous[0], point[1] - previous[1]];
      const outgoing = [following[0] - point[0], following[1] - point[1]];
      return `${incoming[0] * outgoing[1] - incoming[1] * outgoing[0] > 0 ? "left" : "right"}:tight`;
    });
    return { vertices, cornerSequence, sSectionPairs };
  }

  function roundBrowserExtremeRoute(route, width) {
    const anchors = route.vertices, count = anchors.length;
    const incomingPoints = [], outgoingPoints = [];
    const roundness = Math.max(0, Math.min(1, (width - 8) / 92));
    for (let index = 0; index < count; index += 1) {
      const anchor = anchors[index], previous = anchors[(index - 1 + count) % count], following = anchors[(index + 1) % count];
      const incomingLength = trackDistance(anchor, previous), outgoingLength = trackDistance(anchor, following);
      const u = [(previous[0] - anchor[0]) / incomingLength, (previous[1] - anchor[1]) / incomingLength];
      const v = [(following[0] - anchor[0]) / outgoingLength, (following[1] - anchor[1]) / outgoingLength];
      const cosine = Math.max(-1, Math.min(1, u[0] * v[0] + u[1] * v[1]));
      const half = (Math.PI - Math.acos(cosine)) / 2;
      const radius = 1.7 * width * (1 + 0.4 * roundness);
      const trim = Math.min(radius * Math.sin(half) / Math.max(Math.cos(half) ** 2, 1e-9), 0.45 * incomingLength, 0.45 * outgoingLength);
      incomingPoints.push(trackInterpolate(anchor, previous, trim / incomingLength));
      outgoingPoints.push(trackInterpolate(anchor, following, trim / outgoingLength));
    }
    const points = [incomingPoints[0]], ranges = [];
    for (let index = 0; index < count; index += 1) {
      const start = points.length - 1, anchor = anchors[index];
      const controlLength = Math.max(trackDistance(anchor, incomingPoints[index]), trackDistance(anchor, outgoingPoints[index]));
      const steps = Math.max(4, Math.ceil(2 * controlLength / 4));
      for (let step = 1; step <= steps; step += 1) points.push(quadraticBezier(incomingPoints[index], anchor, outgoingPoints[index], step / steps));
      ranges.push([start, points.length - 1]);
      const nextIndex = (index + 1) % count;
      const edgeLength = trackDistance(outgoingPoints[index], incomingPoints[nextIndex]);
      const edgeSteps = Math.max(1, Math.ceil(edgeLength / 4));
      const last = nextIndex ? edgeSteps : edgeSteps - 1;
      for (let step = 1; step <= last; step += 1) points.push(trackInterpolate(outgoingPoints[index], incomingPoints[nextIndex], step / edgeSteps));
    }
    const centerline = points.map(([x, y]) => [roundTrackCoordinate(x), roundTrackCoordinate(y)]);
    if (centerline.length > MAX_CENTERLINE_POINTS) throw new Error(`generated centerline exceeds ${MAX_CENTERLINE_POINTS} points`);
    const measured = measureCornerProfiles(centerline, ranges, width);
    const sSectionConnectorLengths = route.sSectionPairs.map(([first, second]) =>
      trackDistance(outgoingPoints[first].map(roundTrackCoordinate), incomingPoints[second].map(roundTrackCoordinate)));
    const near90CornerCount = measured.turns.filter((turn) => 75 <= Math.abs(turn) && Math.abs(turn) <= 105).length;
    return { centerline, sequence: route.cornerSequence, turns: measured.turns, radii: measured.radii,
      sSectionPairs: route.sSectionPairs, sSectionConnectorLengths, near90CornerCount };
  }

  function validateBrowserExtremeProfile(candidate, width) {
    const count = candidate.sequence.length;
    if (![12, 14, 16].includes(count)) throw new Error(`extreme corner count ${count} is unsupported`);
    if (candidate.sSectionPairs.length < 3) throw new Error("extreme route needs at least three S sections");
    if (candidate.turns.length !== count || candidate.radii.length !== count) throw new Error("extreme corner metadata lengths do not match the route");
    if (candidate.sSectionConnectorLengths.length !== candidate.sSectionPairs.length) throw new Error("extreme S-section metadata lengths do not match the route");
    if (candidate.sSectionConnectorLengths.some((length) => !(0.6 * width - 1e-5 <= length && length <= 2.2 * width + 1e-5))) throw new Error("extreme S-section connector must be 0.6–2.2 track widths");
    for (let index = 0; index < count; index += 1) {
      const [direction, cornerClass] = candidate.sequence[index].split(":", 2);
      if ((direction === "left" ? 1 : -1) * candidate.turns[index] < 20) throw new Error(`measured corner turn does not match ${candidate.sequence[index]}`);
      const [low, high] = CORNER_RADIUS_WIDTH_RANGES[cornerClass];
      if (!(low <= candidate.radii[index] && candidate.radii[index] <= high)) throw new Error(`measured corner radius does not match ${candidate.sequence[index]}`);
    }
    const near90 = candidate.turns.filter((turn) => 75 <= Math.abs(turn) && Math.abs(turn) <= 105).length;
    if (near90 < 3 || near90 !== candidate.near90CornerCount) throw new Error("extreme route needs at least three measured near-90-degree corners");
  }

  function generateBrowserExtremeGeometry(designSeed, width) {
    let lastError;
    for (let attemptIndex = 0; attemptIndex < 64; attemptIndex += 1) {
      try {
        const route = buildBrowserExtremeRoute(designSeed, width, attemptIndex);
        const candidate = roundBrowserExtremeRoute(route, width);
        validateBrowserExtremeProfile(candidate, width);
        validateBrowserCustomGeometry(candidate.centerline, width);
        return candidate;
      } catch (error) { lastError = error; }
    }
    throw new Error(`could not generate a valid extreme_technical track for seed ${designSeed} and width ${width}: ${lastError.message}`);
  }

  function makeBrowserCustomMap(options) {
    if (!options || typeof options !== "object") throw new Error("options must be an object");
    if (typeof options.design_seed !== "number" || !Number.isInteger(options.design_seed) || options.design_seed < 0 || options.design_seed > 4294967295) {
      throw new Error("design_seed must be an integer between 0 and 4294967295");
    }
    const designSeed = options.design_seed;
    if (!TRACK_TEMPLATES.has(options.template)) throw new Error(`template must be one of ${Array.from(TRACK_TEMPLATES).sort().join(", ")}`);
    if (typeof options.width !== "number" || !Number.isFinite(options.width) || options.width < 0.5 || options.width > MAX_GENERATED_TRACK_WIDTH) {
      throw new Error(`width must be between 0.5 and ${MAX_GENERATED_TRACK_WIDTH} for generated tracks`);
    }
    const width = options.width;
    const mapId = String(options.map_id);
    if (!/^custom-track-[A-Za-z0-9][A-Za-z0-9_-]{0,63}$/.test(mapId)) throw new Error("map_id must match custom-track-... ");
    const generated = generateBrowserGeometry(options.template, designSeed, width);
    return {
      schema_version: MAP_SCHEMA_VERSION,
      map_id: mapId,
      map_kind: "custom",
      geometry: { centerline: generated.centerline, width, start_index: 0, direction: 1 },
      obstacle_mode: "custom_only",
      obstacles: Array.isArray(options.obstacles) ? options.obstacles : [],
      max_steps: Math.trunc(finiteNumber(options.max_steps, "max_steps", 1, 10000)),
      frame_skip: Math.trunc(finiteNumber(options.frame_skip, "frame_skip", 1, 16)),
      generator: {
        template: options.template,
        design_seed: designSeed,
        generator_version: 5,
        corner_count: generated.sequence.length,
        corner_sequence: generated.sequence,
        corner_turn_degrees: generated.turns,
        corner_radius_widths: generated.radii,
        ...(options.template === "extreme_technical" ? {
          s_section_count: generated.sSectionPairs.length,
          near_90_corner_count: generated.near90CornerCount
        } : {})
      }
    };
  }

  async function generateCustomMap() {
    const designSeed = integerNumber($("design-seed").value, "디자인 seed", 0, 4294967295);
    const mapId = $("custom-map-id").value.trim();
    const template = $("custom-template").value;
    const width = finiteNumber($("custom-width").value, "도로 반폭", 0.5, MAX_GENERATED_TRACK_WIDTH);
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
    updateAgentPicker();
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
      if (Array.isArray(step.position) && step.position.length === 2) {
        state.manualVehicle = {
          position: step.position.map(Number),
          angle: Number(step.angle || 0)
        };
        drawTrack($("map-canvas"), state.preview, state.mapSpec.obstacles, "canvas-empty", state.manualVehicle);
      }
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
    const agentId = policy === "agent" ? $("agent-select").value : null;
    if (policy === "agent" && !agentId) throw new Error("실행할 Agent를 선택하세요.");
    if (policy === "agent" && !state.agents.some((agent) => agent.id === agentId && agent.ready)) {
      throw new Error("선택한 Agent에 model.pt 또는 policy.pt가 없습니다.");
    }
    if (policy === "manual") {
      const started = await apiRequest("/api/runs/start", {
        method: "POST",
        body: JSON.stringify({ map, policy: "manual", record_frames: recordFrames })
      });
      state.manualRunId = started.run_id;
      state.preview = { track: started.track, official_obstacles: [] };
      state.manualVehicle = null;
      state.manualPaused = false;
      state.manualKeys.clear();
      setManualControls(true);
      updateManualActionStatus();
      renderMap();
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
        body: JSON.stringify({ map, policy, agent_id: agentId, record_frames: recordFrames }),
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
    $("run-policy").addEventListener("change", () => updateAgentPicker());
    $("agent-select").addEventListener("change", (event) => {
      state.selectedAgentId = event.target.value;
    });
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
    drawTrack($("map-canvas"), state.preview, state.mapSpec.obstacles, "canvas-empty", state.manualVehicle);
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
    updateGeneratedTrackSummary(state.mapSpec);
    renderComparison();
    checkLocalApi().then((available) => {
      if (!available) setStatus("파일 보기 모드입니다. 로그 재생과 맵 JSON 저장은 계속 사용할 수 있습니다.", false);
    });
  }

  window.HAICSimulator = {
    loadRunLog,
    normalizeMap,
    drawTrack,
    renderRunFrame,
    setPlaybackRunning,
    advancePlayback,
    summarizeRuns,
    renderComparison,
    generateCustomMap,
    applyCustomMap,
    startRun,
    sendManualAction,
    finishRun,
    makeBrowserCustomMap,
    formatGeneratedTrackSummary,
    manualActionFromKeys,
    previewFromCustomMap,
    customMapValidation
  };
  document.addEventListener("DOMContentLoaded", initialize);
}());

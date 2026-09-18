(function () {
  "use strict";

  const SCHEMA_VERSION = 1;
  const DEFAULT_MAP = {
    schema_version: SCHEMA_VERSION,
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
    runState: null
  };

  const $ = (id) => document.getElementById(id);

  function setStatus(message, isError) {
    const target = $("file-status");
    if (target) {
      target.textContent = message;
      target.classList.toggle("error", Boolean(isError));
    }
  }

  function finiteNumber(value, name, minimum, maximum) {
    const number = Number(value);
    if (!Number.isFinite(number) || number < minimum || number > maximum) {
      throw new Error(`${name}은 ${minimum}에서 ${maximum} 사이여야 합니다.`);
    }
    return number;
  }

  function normalizeMap(payload) {
    if (!payload || typeof payload !== "object") throw new Error("맵 JSON은 객체여야 합니다.");
    if (payload.schema_version !== SCHEMA_VERSION) throw new Error("지원하지 않는 map schema version입니다.");
    const obstacles = Array.isArray(payload.obstacles) ? payload.obstacles : [];
    if (obstacles.length > 64) throw new Error("사용자 장애물은 최대 64개입니다.");
    const normalized = obstacles.map((obstacle, index) => {
      if (!obstacle || typeof obstacle !== "object") throw new Error(`obstacles[${index}] 형식이 잘못되었습니다.`);
      return {
        progress: finiteNumber(obstacle.progress, "progress", 0, 1),
        lateral: finiteNumber(obstacle.lateral, "lateral", -1, 1),
        radius: finiteNumber(obstacle.radius, "radius", 0.2, 4)
      };
    });
    const mode = payload.obstacle_mode || "official";
    if (!["official", "custom_only", "official_plus_custom"].includes(mode)) throw new Error("장애물 모드가 잘못되었습니다.");
    return {
      schema_version: SCHEMA_VERSION,
      track_id: Math.trunc(finiteNumber(payload.track_id, "track_id", 1, Number.MAX_SAFE_INTEGER)),
      seed: Math.trunc(finiteNumber(payload.seed, "seed", 0, 4294967295)),
      obstacle_mode: mode,
      obstacles: normalized,
      max_steps: Math.trunc(finiteNumber(payload.max_steps ?? 2000, "max_steps", 1, 10000)),
      frame_skip: Math.trunc(finiteNumber(payload.frame_skip ?? 4, "frame_skip", 1, 16))
    };
  }

  function readForm() {
    state.mapSpec = normalizeMap({
      ...state.mapSpec,
      track_id: $("track-id").value,
      seed: $("seed").value,
      obstacle_mode: $("obstacle-mode").value,
      max_steps: $("max-steps").value,
      frame_skip: $("frame-skip").value,
      obstacles: state.mapSpec.obstacles
    });
    return state.mapSpec;
  }

  function syncForm() {
    const map = state.mapSpec;
    $("track-id").value = map.track_id;
    $("seed").value = map.seed;
    $("obstacle-mode").value = map.obstacle_mode;
    $("max-steps").value = map.max_steps;
    $("frame-skip").value = map.frame_skip;
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

  function drawTrack(canvas, preview, obstacles) {
    const context = canvas.getContext("2d");
    const width = canvas.width;
    const height = canvas.height;
    context.clearRect(0, 0, width, height);
    context.fillStyle = "#080b10";
    context.fillRect(0, 0, width, height);
    if (!preview || !preview.track || !Array.isArray(preview.track.points) || !preview.track.points.length) {
      $("canvas-empty").classList.remove("hidden");
      return;
    }
    $("canvas-empty").classList.add("hidden");
    const points = preview.track.points;
    const bounds = worldBounds(points);
    const mapped = points.map((point) => project(point, bounds, width, height));
    context.lineJoin = "round";
    context.lineCap = "round";
    context.beginPath();
    mapped.forEach((point, index) => index ? context.lineTo(point.x, point.y) : context.moveTo(point.x, point.y));
    context.strokeStyle = "#394556";
    context.lineWidth = Math.max(22, Math.min(48, preview.track.width * 2.2));
    context.stroke();
    context.beginPath();
    mapped.forEach((point, index) => index ? context.lineTo(point.x, point.y) : context.moveTo(point.x, point.y));
    context.strokeStyle = "#8c98a9";
    context.lineWidth = 2;
    context.setLineDash([8, 8]);
    context.stroke();
    context.setLineDash([]);
    const drawObstacle = (position, radius, color) => {
      const point = project(position, bounds, width, height);
      const scale = Math.min(width / (bounds.maxX - bounds.minX), height / (bounds.maxY - bounds.minY));
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
    drawTrack($("map-canvas"), state.preview, state.mapSpec.obstacles);
    $("map-status").textContent = `${state.mapSpec.obstacles.length}개 사용자 장애물`;
  }

  function applyMap(payload, source) {
    state.mapSpec = normalizeMap(payload);
    state.preview = payload.preview || null;
    syncForm();
    renderMap();
    setStatus(`${source}을 불러왔습니다.`, false);
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

  function generateRandomObstacles() {
    readForm();
    const count = Math.trunc(finiteNumber($("random-count").value, "개수", 1, 64));
    const random = seededRandom(state.mapSpec.seed ^ 0xc0de);
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
    readForm();
    const payload = { ...state.mapSpec };
    const blob = new Blob([JSON.stringify(payload, null, 2)], { type: "application/json" });
    const link = document.createElement("a");
    link.href = URL.createObjectURL(blob);
    link.download = "map.json";
    link.click();
    URL.revokeObjectURL(link.href);
    setStatus("편집한 map.json을 다운로드했습니다.", false);
  }

  function readJsonFile(file, callback) {
    const reader = new FileReader();
    reader.onload = () => {
      try { callback(JSON.parse(reader.result)); } catch (error) { setStatus(`JSON을 읽지 못했습니다: ${error.message}`, true); }
    };
    reader.onerror = () => setStatus("파일을 읽지 못했습니다.", true);
    reader.readAsText(file);
  }

  function bindEvents() {
    $("map-file").addEventListener("change", (event) => {
      const file = event.target.files[0];
      if (file) readJsonFile(file, (payload) => applyMap(payload, file.name));
    });
    $("log-file").addEventListener("change", (event) => {
      const file = event.target.files[0];
      if (file) setStatus(`${file.name}은 다음 단계에서 재생 로그로 불러옵니다.`, false);
    });
    $("add-obstacle").addEventListener("click", () => {
      try { addObstacleFromForm(); } catch (error) { setStatus(error.message, true); }
    });
    $("random-obstacles").addEventListener("click", () => {
      try { generateRandomObstacles(); } catch (error) { setStatus(error.message, true); }
    });
    $("clear-obstacles").addEventListener("click", () => { state.mapSpec.obstacles = []; renderMap(); setStatus("사용자 장애물을 모두 삭제했습니다.", false); });
    $("download-map").addEventListener("click", () => { try { downloadMap(); } catch (error) { setStatus(error.message, true); } });
    $("map-form").addEventListener("change", () => { try { readForm(); renderMap(); } catch (error) { setStatus(error.message, true); } });
    $("obstacle-table").addEventListener("input", (event) => {
      const input = event.target.closest("[data-obstacle]");
      if (!input) return;
      try {
        const index = Number(input.dataset.obstacle);
        const key = input.dataset.key;
        const limits = { progress: [0, 1], lateral: [-1, 1], radius: [.2, 4] }[key];
        state.mapSpec.obstacles[index][key] = finiteNumber(input.value, key, limits[0], limits[1]);
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
      if (file) readJsonFile(file, (payload) => applyMap(payload, file.name));
    });
  }

  function initialize() {
    bindEvents();
    syncForm();
    renderMap();
  }

  document.addEventListener("DOMContentLoaded", initialize);
}());

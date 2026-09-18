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

  function normalizeRun(payload) {
    if (!payload || typeof payload !== "object") throw new Error("실행 로그는 객체여야 합니다.");
    if (payload.schema_version !== SCHEMA_VERSION) throw new Error("지원하지 않는 run schema version입니다.");
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
    const source = state.preview || {};
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
    $("metric-track").textContent = `${map.track_id ?? "—"} / ${map.seed ?? "—"}`;
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
      if (file) readJsonFile(file, (payload) => addRunPayload(payload, file.name));
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
    syncForm();
    renderMap();
    renderComparison();
  }

  window.HAICSimulator = { loadRunLog, renderRunFrame, setPlaybackRunning, advancePlayback, summarizeRuns };
  document.addEventListener("DOMContentLoaded", initialize);
}());

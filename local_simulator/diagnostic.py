"""Run an agent and produce a camera/trajectory diagnostic report."""

from __future__ import annotations

import argparse
import base64
import html
import json
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from .logging import load_run_log, run_log_to_dict, save_run_log
from .policies import AgentPolicy
from .schema import map_from_dict
from .simulation import run_episode


def _load_map(path: Path):
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError(f"could not read map JSON: {path}") from error
    return map_from_dict(payload)


def _frame_bytes(encoded: str) -> np.ndarray:
    try:
        raw = base64.b64decode(encoded, validate=True)
    except (ValueError, TypeError) as error:
        raise ValueError("run log contains an invalid camera frame") from error
    frame = cv2.imdecode(np.frombuffer(raw, dtype=np.uint8), cv2.IMREAD_COLOR)
    if frame is None or frame.size == 0:
        raise ValueError("run log contains an unreadable camera frame")
    return frame


def write_mp4(run_log, output_path: str | Path, *, frame_stride: int = 1) -> Path:
    """Write recorded camera frames to an MP4 without rerunning the agent."""

    if frame_stride < 1:
        raise ValueError("frame_stride must be positive")
    if not run_log.frames:
        raise ValueError("run log has no camera frames; rerun with --record-frames")
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    selected = run_log.frames[::frame_stride]
    first = _frame_bytes(selected[0])
    height, width = first.shape[:2]
    fps = float(run_log.run.get("control_hz", 12.5)) / frame_stride
    if not np.isfinite(fps) or fps <= 0:
        fps = 12.5 / frame_stride
    writer = cv2.VideoWriter(
        str(path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height)
    )
    if not writer.isOpened():
        raise RuntimeError("OpenCV could not open an MP4 writer on this machine")
    try:
        for encoded in selected:
            frame = _frame_bytes(encoded)
            if frame.shape[:2] != (height, width):
                frame = cv2.resize(frame, (width, height), interpolation=cv2.INTER_AREA)
            writer.write(frame)
    finally:
        writer.release()
    if not path.is_file() or path.stat().st_size == 0:
        raise RuntimeError(f"MP4 writer produced no output: {path}")
    return path


def _visual_payload(run_log, *, frame_stride: int) -> dict[str, Any]:
    if frame_stride < 1:
        raise ValueError("frame_stride must be positive")
    payload = run_log_to_dict(run_log)
    steps = payload["steps"]
    frames = payload.get("frames") or []
    frame_indices = list(range(0, len(frames), frame_stride))
    return {
        "run": payload["run"],
        "track": payload["track"],
        "steps": steps,
        "summary": payload["summary"],
        "frames": [frames[index] for index in frame_indices],
        "frame_steps": frame_indices,
    }


def _html_document(payload: dict[str, Any]) -> str:
    # Prevent a camera payload or metadata string from closing the JSON script.
    encoded = json.dumps(payload, ensure_ascii=False, separators=(",", ":"))
    encoded = encoded.replace("</", "<\\/")
    title = html.escape(str(payload["run"].get("policy", {}).get("name", "Agent")))
    return f"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>HAIC agent diagnostic — {title}</title>
<style>
:root {{ color-scheme: dark; --bg:#101318; --panel:#191e26; --line:#303846; --text:#edf1f7; --muted:#9ca8b8; --accent:#6dd6c0; --warn:#f3b562; --bad:#f47c8c; }}
* {{ box-sizing:border-box; }}
body {{ margin:0; background:var(--bg); color:var(--text); font:14px/1.4 system-ui, sans-serif; }}
main {{ max-width:1180px; margin:0 auto; padding:24px; }}
h1 {{ margin:0 0 16px; font-size:22px; }}
h2 {{ margin:0 0 10px; font-size:15px; color:var(--muted); }}
.layout {{ display:grid; grid-template-columns:minmax(0,1.25fr) minmax(280px,.75fr); gap:16px; }}
.panel {{ background:var(--panel); border:1px solid var(--line); border-radius:10px; padding:14px; }}
canvas {{ width:100%; height:auto; display:block; background:#0d1117; border-radius:6px; }}
#camera {{ width:100%; aspect-ratio:4/3; object-fit:contain; background:#0d1117; border-radius:6px; }}
.controls {{ display:flex; gap:8px; align-items:center; flex-wrap:wrap; margin-top:12px; }}
button, select, input[type=range] {{ accent-color:var(--accent); }}
button {{ color:var(--text); background:#252d39; border:1px solid var(--line); border-radius:6px; padding:7px 11px; cursor:pointer; }}
button:hover {{ border-color:var(--accent); }}
input[type=range] {{ flex:1; min-width:160px; }}
.stats {{ display:grid; grid-template-columns:repeat(3,1fr); gap:8px; margin-top:12px; }}
.stat {{ border:1px solid var(--line); border-radius:6px; padding:8px; }}
.stat span {{ display:block; color:var(--muted); font-size:12px; }}
.stat strong {{ display:block; margin-top:2px; font-size:17px; }}
.legend {{ color:var(--muted); font-size:12px; margin-top:8px; }}
.legend span {{ margin-right:12px; }}
.swatch {{ display:inline-block; width:10px; height:3px; vertical-align:middle; margin-right:4px; }}
#flags {{ margin:0; padding-left:18px; color:var(--muted); }}
#flags li {{ margin:4px 0; }}
.bad {{ color:var(--bad); }} .warn {{ color:var(--warn); }}
@media (max-width:760px) {{ main {{ padding:12px; }} .layout {{ grid-template-columns:1fr; }} .stats {{ grid-template-columns:repeat(2,1fr); }} }}
</style>
</head>
<body>
<main>
<h1>HAIC agent diagnostic replay</h1>
<div class="layout">
<section class="panel">
<h2>Trajectory and current action</h2>
<canvas id="track" width="820" height="560" aria-label="Agent trajectory"></canvas>
<div class="legend"><span><i class="swatch" style="background:#6dd6c0"></i>trajectory</span><span><i class="swatch" style="background:#f47c8c"></i>collision</span><span><i class="swatch" style="background:#f3b562"></i>current step</span></div>
<canvas id="signals" width="820" height="230" aria-label="Actions, progress, and damage"></canvas>
</section>
<section class="panel">
<h2>Camera frame</h2>
<img id="camera" alt="Camera frame at selected step">
<div class="controls">
<button id="play" type="button">Play</button><button id="pause" type="button">Pause</button>
<label>speed <select id="speed"><option value="0.5">0.5×</option><option value="1" selected>1×</option><option value="2">2×</option><option value="4">4×</option></select></label>
</div>
<div class="controls"><span id="step-label">step 0</span><input id="step" type="range" min="0" max="0" value="0" aria-label="Replay step"><span id="step-total">/ 0</span></div>
<div class="stats"><div class="stat"><span>progress</span><strong id="progress">—</strong></div><div class="stat"><span>damage</span><strong id="damage">—</strong></div><div class="stat"><span>collisions</span><strong id="collisions">—</strong></div><div class="stat"><span>steer</span><strong id="steer">—</strong></div><div class="stat"><span>gas</span><strong id="gas">—</strong></div><div class="stat"><span>brake</span><strong id="brake">—</strong></div></div>
<h2 style="margin-top:16px">Observed warning signals</h2>
<ul id="flags" aria-live="polite"></ul>
</section>
</div>
</main>
<script id="run-data" type="application/json">{encoded}</script>
<script>
(() => {{
  const data = JSON.parse(document.getElementById('run-data').textContent);
  const steps = data.steps || [], frames = data.frames || [], frameSteps = data.frame_steps || [];
  const stepInput = document.getElementById('step');
  const camera = document.getElementById('camera');
  const trackCanvas = document.getElementById('track'), signalCanvas = document.getElementById('signals');
  const state = {{ index: 0, playing: false, timer: null }};
  stepInput.max = Math.max(0, steps.length - 1); document.getElementById('step-total').textContent = '/ ' + steps.length;
  const pct = (value) => (100 * Number(value || 0)).toFixed(1) + '%';
  const fmt = (value) => Number(value || 0).toFixed(3);
  function bounds(points) {{
    const xs = points.map(p => p[2]), ys = points.map(p => p[3]);
    const minX = Math.min(...xs), maxX = Math.max(...xs), minY = Math.min(...ys), maxY = Math.max(...ys);
    return {{ minX, maxX, minY, maxY, sx: (v) => 26 + (v - minX) / Math.max(1e-6, maxX - minX) * 768, sy: (v) => 520 - (v - minY) / Math.max(1e-6, maxY - minY) * 480 }};
  }}
  function drawTrack() {{
    const ctx = trackCanvas.getContext('2d'), points = data.track.points || [], b = bounds(points);
    ctx.clearRect(0, 0, trackCanvas.width, trackCanvas.height); ctx.lineWidth = 14; ctx.strokeStyle = '#303846'; ctx.lineCap = 'round';
    ctx.beginPath(); points.forEach((p, i) => i ? ctx.lineTo(b.sx(p[2]), b.sy(p[3])) : ctx.moveTo(b.sx(p[2]), b.sy(p[3]))); ctx.stroke();
    ctx.lineWidth = 2; ctx.strokeStyle = '#6dd6c0'; ctx.beginPath();
    steps.slice(0, state.index + 1).forEach((s, i) => i ? ctx.lineTo(b.sx(s.position[0]), b.sy(s.position[1])) : ctx.moveTo(b.sx(s.position[0]), b.sy(s.position[1]))); ctx.stroke();
    steps.forEach((s) => {{ if (s.collision) {{ ctx.fillStyle = '#f47c8c'; ctx.beginPath(); ctx.arc(b.sx(s.position[0]), b.sy(s.position[1]), 4, 0, Math.PI * 2); ctx.fill(); }} }});
    const current = steps[state.index]; if (current) {{ ctx.fillStyle = '#f3b562'; ctx.beginPath(); ctx.arc(b.sx(current.position[0]), b.sy(current.position[1]), 6, 0, Math.PI * 2); ctx.fill(); }}
  }}
  function drawSignals() {{
    const ctx = signalCanvas.getContext('2d'), w = signalCanvas.width, h = signalCanvas.height; ctx.clearRect(0, 0, w, h);
    const x = (i) => 20 + i / Math.max(1, steps.length - 1) * (w - 36), y = (v, lo, hi, top, bottom) => bottom - (v - lo) / (hi - lo) * (bottom - top);
    ctx.strokeStyle = '#303846'; ctx.lineWidth = 1; [0, .5, 1].forEach(v => {{ const yy = y(v, 0, 1, 18, h - 20); ctx.beginPath(); ctx.moveTo(20, yy); ctx.lineTo(w - 16, yy); ctx.stroke(); }});
    const line = (key, color, lo, hi, top, bottom) => {{ ctx.strokeStyle = color; ctx.lineWidth = 2; ctx.beginPath(); steps.forEach((s, i) => i ? ctx.lineTo(x(i), y(Number(s[key]), lo, hi, top, bottom)) : ctx.moveTo(x(i), y(Number(s[key]), lo, hi, top, bottom))); ctx.stroke(); }};
    line('progress', '#6dd6c0', 0, 1, 18, h - 20); line('damage', '#f47c8c', 0, 1, 18, h - 20);
    const actionLine = (getter, color, strokeWidth) => {{ ctx.strokeStyle = color; ctx.lineWidth = strokeWidth; ctx.beginPath(); steps.forEach((s, i) => {{ const v = getter(s); i ? ctx.lineTo(x(i), y(v, -1, 1, 18, h - 20)) : ctx.moveTo(x(i), y(v, -1, 1, 18, h - 20)); }}); ctx.stroke(); }};
    actionLine((s) => Number(s.action?.[0] || 0), '#f3b562', 2);
    actionLine((s) => Number(s.action?.[1] || 0) - Number(s.action?.[2] || 0), '#c4ccda', 1);
    ctx.fillStyle = '#9ca8b8'; ctx.font = '12px system-ui'; ctx.fillText('progress / damage / steer / gas-brake', 20, 14);
    const currentX = x(state.index); ctx.strokeStyle = '#f3b562'; ctx.beginPath(); ctx.moveTo(currentX, 0); ctx.lineTo(currentX, h); ctx.stroke();
  }}
  function warnings() {{
    const flags = [], saturated = steps.filter(s => Math.abs(Number(s.action?.[0] || 0)) >= .95).length;
    const overlap = steps.filter(s => Number(s.action?.[1] || 0) > .05 && Number(s.action?.[2] || 0) > .05).length;
    const collisions = steps.filter(s => s.collision).length;
    if (saturated) flags.push(['warn', `steering saturation: ${{saturated}} steps`]);
    if (overlap) flags.push(['warn', `gas and brake overlap: ${{overlap}} steps`]);
    if (collisions) flags.push(['bad', `collision events: ${{collisions}}`]);
    if (!flags.length) flags.push(['', 'no rule-based warning signal found']);
    document.getElementById('flags').innerHTML = flags.map(([klass, text]) => `<li class="${{klass}}">${{text}}</li>`).join('');
  }}
  function frameForStep(index) {{
    let selected = -1; frameSteps.forEach((step, i) => {{ if (step <= index) selected = i; }});
    if (selected >= 0) camera.src = 'data:image/jpeg;base64,' + frames[selected]; else camera.removeAttribute('src');
  }}
  function render() {{
    const current = steps[state.index] || {{ action: [0, 0, 0], progress: 0, damage: 0 }};
    stepInput.value = state.index; document.getElementById('step-label').textContent = 'step ' + state.index;
    document.getElementById('progress').textContent = pct(current.progress); document.getElementById('damage').textContent = pct(current.damage);
    document.getElementById('collisions').textContent = String((data.summary || {{}}).collision_count ?? '—');
    document.getElementById('steer').textContent = fmt(current.action?.[0]); document.getElementById('gas').textContent = fmt(current.action?.[1]); document.getElementById('brake').textContent = fmt(current.action?.[2]);
    frameForStep(state.index); drawTrack(); drawSignals();
  }}
  function tick() {{ if (!state.playing) return; if (state.index >= steps.length - 1) {{ state.playing = false; return; }} state.index += 1; render(); state.timer = setTimeout(tick, 1000 / (Number(document.getElementById('speed').value) * 12.5)); }}
  document.getElementById('play').onclick = () => {{ state.playing = true; tick(); }}; document.getElementById('pause').onclick = () => {{ state.playing = false; clearTimeout(state.timer); }};
  stepInput.oninput = () => {{ state.playing = false; state.index = Number(stepInput.value); render(); }};
  warnings(); render();
}})();
</script>
</body></html>"""


def write_report(run_log, output_path: str | Path, *, frame_stride: int = 1) -> Path:
    """Write a standalone HTML replay and diagnostic report."""

    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(_html_document(_visual_payload(run_log, frame_stride=frame_stride)), encoding="utf-8")
    return path


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--map", type=Path, help="map JSON; runs the agent and records camera frames")
    source.add_argument("--run-log", type=Path, help="existing run JSON; skips simulation")
    parser.add_argument("--project-root", type=Path, default=Path.cwd())
    parser.add_argument("--agent-path", type=Path)
    parser.add_argument("--output-dir", type=Path, default=Path(".haic-artifacts/diagnostics"))
    parser.add_argument("--frame-stride", type=int, default=1)
    parser.add_argument("--video", action="store_true", help="also write an MP4 from recorded frames")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    if args.frame_stride < 1:
        _build_parser().error("--frame-stride must be positive")
    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    if args.map is not None:
        document = _load_map(args.map)
        project_root = args.project_root.resolve()
        agent_path = (args.agent_path or project_root / "agent.py").resolve()
        run_log = run_episode(
            document,
            AgentPolicy(agent_path, project_root),
            record_frames=True,
        )
        run_path = output_dir / "run.json"
        save_run_log(run_log, run_path)
    else:
        run_path = args.run_log.resolve()
        run_log = load_run_log(run_path)
    report_path = write_report(run_log, output_dir / "report.html", frame_stride=args.frame_stride)
    print(f"run: {run_path}")
    print(f"report: {report_path}")
    if args.video:
        video_path = write_mp4(run_log, output_dir / "replay.mp4", frame_stride=args.frame_stride)
        print(f"video: {video_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

> **Completed historical implementation plan, relocated 2026-09-23.** Its unchecked
> checklist reflects the original planning record rather than the resulting code.
> Completion here means the implementation work is no longer active; use
> `docs/architecture/overview.md` and code/tests for current behavior.

# 로컬 시뮬레이터 및 로그 재생 사이트 구현 계획

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 공식 HAIC 물리 환경으로 맵을 만들고 로컬 주행 로그를 생성한 뒤, 정적 웹사이트에서 그 로그를 업로드·재생·비교할 수 있게 한다.

**Architecture:** Python 로컬 실행기가 `map.json`을 읽어 공식 `CarRacing`/`CarEnvironment`를 실행하고 `run.json`을 출력한다. 정적 HTML/CSS/JavaScript 사이트는 맵과 로그를 브라우저에서 읽어 미니맵 궤적, 선택적 카메라 프레임, 기록을 재생하며 서버 측 물리 세션은 사용하지 않는다. 이후 에이전트 제작 단계에서는 동일한 실행기의 policy adapter만 `Agent.act()`로 교체한다.

**Tech Stack:** Python 3.11, Gymnasium 0.29.1, Box2D, NumPy 1.26, OpenCV 4.8, Python `unittest`, dependency-free HTML/CSS/JavaScript Canvas.

**Spec:** `../../architecture/history/2026-09-18-obstacle-simulation-site-design.md`

## Global Constraints

- 공식 `core/`, `env_wrapper.py`, `damage.py`, `agent.py`는 수정하지 않는다.
- 공식 제어 규칙은 `frame_skip=4`, `no_operation=50`, `max_steps * frame_skip + 200` raw-step budget을 유지한다.
- 맵과 로그 schema version은 `1`로 시작하고 필수 필드는 명시적으로 검증한다.
- 브라우저는 Box2D를 실행하지 않고 로그를 재생만 한다.
- 새 Python dependency를 추가하지 않는다.
- 대용량 로그·프레임·체크포인트는 `D:\HAIC`에 저장하고 저장소에는 작은 fixture만 둔다.
- 각 작업은 실패 테스트 → 최소 구현 → 관련 테스트 → 커밋 순서로 진행한다.

---

### Task 1: Map and obstacle schema

**Files:**
- Create: `local_simulator/__init__.py`
- Create: `local_simulator/schema.py`
- Create: `tests/test_local_simulator_schema.py`

**Interfaces:**
- `CustomObstacle(progress: float, lateral: float, radius: float)` dataclass
- `MapSpec(track_id: int, seed: int, obstacle_mode: str, obstacles: tuple[CustomObstacle, ...], max_steps: int, frame_skip: int, schema_version: int = 1)` dataclass
- `validate_map_payload(payload: object) -> MapSpec`
- `map_to_dict(spec: MapSpec) -> dict[str, Any]`
- `map_from_dict(payload: object) -> MapSpec`

**Validation rules:**
- `track_id` is a positive integer.
- `seed` is an integer in `0..4294967295`.
- `obstacle_mode` is one of `official`, `custom_only`, `official_plus_custom`.
- `progress` is in `[0.0, 1.0]`, `lateral` is in `[-1.0, 1.0]`, and `radius` is in `[0.2, 4.0]`.
- At most 64 custom obstacles are accepted.
- `max_steps` is in `[1, 10000]` and `frame_skip` is in `[1, 16]`.
- Unknown optional fields are preserved only under a separate `metadata` field; malformed required fields raise `ValueError`.

- [ ] **Step 1: Write failing schema tests**

```python
class TestMapSchema(unittest.TestCase):
    def test_round_trip_preserves_map(self):
        spec = MapSpec(
            track_id=2,
            seed=123,
            obstacle_mode="official_plus_custom",
            obstacles=(CustomObstacle(0.42, -0.25, 1.2),),
            max_steps=2000,
            frame_skip=4,
        )
        self.assertEqual(map_from_dict(map_to_dict(spec)), spec)

    def test_rejects_invalid_seed_and_obstacle(self):
        with self.assertRaises(ValueError):
            validate_map_payload({"track_id": 1, "seed": -1})
        with self.assertRaises(ValueError):
            validate_map_payload({
                "track_id": 1,
                "seed": 1,
                "obstacle_mode": "official",
                "obstacles": [{"progress": 1.2, "lateral": 0, "radius": 1.2}],
            })
```

- [ ] **Step 2: Run the focused test and verify it fails**

Run: `D:\HAIC\haic-env\Scripts\python.exe -m unittest tests.test_local_simulator_schema -v`

Expected: FAIL because `local_simulator.schema` does not exist.

- [ ] **Step 3: Implement the schema and validation**

Use frozen dataclasses, convert JSON lists to tuples at the boundary, and return deterministic dictionaries with `schema_version`, `track_id`, `seed`, `obstacle_mode`, `obstacles`, `max_steps`, and `frame_skip`.

- [ ] **Step 4: Run focused and baseline tests**

Run: `D:\HAIC\haic-env\Scripts\python.exe -m unittest tests.test_local_simulator_schema tests.test_local_contract -v`

Expected: all focused and existing contract tests pass.

- [ ] **Step 5: Commit**

```text
git add local_simulator tests/test_local_simulator_schema.py
git commit -m "feat: add map and obstacle schema"
```

### Task 2: Official environment setup and obstacle placement

**Files:**
- Create: `local_simulator/environment.py`
- Create: `tests/test_local_simulator_environment.py`

**Interfaces:**
- `TrackSnapshot(points: tuple[tuple[float, float, float, float], ...], width: float)` dataclass
- `MapBundle(spec: MapSpec, track: TrackSnapshot, official_obstacles: tuple[tuple[float, float], ...], custom_obstacles: tuple[ObstacleSpec, ...])` dataclass
- `create_environment(spec: MapSpec, render_mode: str | None = "rgb_array") -> tuple[CarEnvironment, CarRacing]`
- `snapshot_track(environment: CarEnvironment) -> TrackSnapshot`
- `map_custom_obstacles(environment: CarEnvironment, spec: MapSpec) -> tuple[ObstacleSpec, ...]`
- `attach_custom_obstacles(environment: CarEnvironment, obstacles: tuple[ObstacleSpec, ...]) -> None`
- `official_obstacle_positions(environment: CarEnvironment) -> list[tuple[float, float]]`
- `build_map_bundle(spec: MapSpec) -> MapBundle`

`ObstacleSpec` is the frozen dataclass already defined in `core.track_variables`; the new module imports it without modifying the official file.

`create_environment` must construct `CarRacing(continuous=True, render_mode=render_mode)` inside `TimeLimit`, wrap it with `CarEnvironment(skip_frames=spec.frame_skip)`, and apply the obstacle mode. For `official`, pass `options={"track_id": spec.track_id}`. For `custom_only`, omit `track_id` so official obstacles are not created. For `official_plus_custom`, keep the official six obstacles and attach the custom bodies after `CarEnvironment.reset()` completes.

Convert each normalized custom obstacle by selecting the nearest track point at `round(progress * (len(track) - 1))`, using its `beta` direction, and applying a bounded lateral offset. Construct `ObstacleSpec` values and call the existing environment obstacle creation path so `ObstacleUserData` and collision damage remain active.

- [ ] **Step 1: Write failing deterministic and placement tests**

```python
class TestEnvironmentSetup(unittest.TestCase):
    def test_same_map_has_same_track_snapshot(self):
        spec = MapSpec(1, 42, "official_plus_custom", (), 20, 4)
        first, _ = create_environment(spec, render_mode=None)
        second, _ = create_environment(spec, render_mode=None)
        try:
            first.reset(seed=spec.seed, options={"track_id": spec.track_id})
            second.reset(seed=spec.seed, options={"track_id": spec.track_id})
            self.assertEqual(snapshot_track(first), snapshot_track(second))
        finally:
            first.close()
            second.close()

    def test_custom_obstacle_is_attached_and_counted(self):
        spec = MapSpec(
            1, 42, "custom_only",
            (CustomObstacle(0.50, 0.0, 1.2),), 20, 4,
        )
        environment, _ = create_environment(spec, render_mode=None)
        try:
            environment.reset(seed=spec.seed)
            custom = map_custom_obstacles(environment, spec)
            attach_custom_obstacles(environment, custom)
            self.assertEqual(len(environment.unwrapped.obstacles), 1)
        finally:
            environment.close()
```

- [ ] **Step 2: Run the focused test and verify it fails**

Run: `D:\HAIC\haic-env\Scripts\python.exe -m unittest tests.test_local_simulator_environment -v`

Expected: FAIL because the environment helper functions do not exist.

- [ ] **Step 3: Implement official environment creation and coordinate conversion**

Keep all imports of `core.vendor` in this new module. Do not edit vendored files. Ensure custom bodies are added only after the wrapper reset has attached the contact listener and initialized damage.

- [ ] **Step 4: Run focused and official contract tests**

Run: `D:\HAIC\haic-env\Scripts\python.exe -m unittest tests.test_local_simulator_environment tests.test_local_contract -v`

Expected: all tests pass; the server parity test remains skipped only because the server repository is absent.

- [ ] **Step 5: Commit**

```text
git add local_simulator/environment.py tests/test_local_simulator_environment.py
git commit -m "feat: add official environment and custom obstacles"
```

### Task 3: Run logs and policy adapter

**Files:**
- Create: `local_simulator/policies.py`
- Create: `local_simulator/logging.py`
- Create: `local_simulator/simulation.py`
- Create: `tests/test_local_simulator_run.py`

**Interfaces:**
- `Policy` protocol with `reset(observation: np.ndarray) -> None` and `act(observation: np.ndarray) -> np.ndarray`
- `BaselinePolicy.act(observation) -> np.ndarray`, returning `[0.0, 1.0, 0.0]`
- `RunStep` dataclass with `step`, `sim_time_s`, `action`, `position`, `angle`, `velocity`, `progress`, `damage`, `collision`, `terminated`, and `truncated`
- `RunLog` dataclass with `schema_version`, `run`, `track`, `steps`, `summary`, and optional `frames`
- `run_episode(spec: MapSpec, policy: Policy, record_frames: bool = False) -> RunLog`
- `save_run_log(run_log: RunLog, output_path: Path) -> None`
- `load_run_log(input_path: Path) -> RunLog`

`run_episode` must reset the wrapper with the map seed and official track option when applicable, attach custom obstacles, call the policy once per wrapper step, clip/validate actions to the official bounds, and stop at `terminated`, `truncated`, or `spec.max_steps`. Each step records the raw car position, angle, velocity, progress and damage from the unwrapped environment and wrapper info. When `record_frames=True`, encode each RGB frame as JPEG bytes and store base64 strings under `frames`; otherwise omit them.

- [ ] **Step 1: Write failing run and log tests**

```python
class TestRunLog(unittest.TestCase):
    def test_baseline_run_has_schema_and_steps(self):
        spec = MapSpec(1, 42, "custom_only", (), 2, 4)
        result = run_episode(spec, BaselinePolicy())
        self.assertEqual(result.schema_version, 1)
        self.assertEqual(len(result.steps), 2)
        self.assertEqual(result.steps[0].action, (0.0, 1.0, 0.0))

    def test_log_round_trip(self):
        spec = MapSpec(1, 42, "custom_only", (), 1, 4)
        result = run_episode(spec, BaselinePolicy())
        with tempfile.TemporaryDirectory() as root:
            path = Path(root) / "run.json"
            save_run_log(result, path)
            loaded = load_run_log(path)
            self.assertEqual(loaded.summary, result.summary)
            self.assertEqual(loaded.steps, result.steps)
```

The test imports `tempfile` and `Path`; the production API remains `Path` based.

- [ ] **Step 2: Run the focused test and verify it fails**

Run: `D:\HAIC\haic-env\Scripts\python.exe -m unittest tests.test_local_simulator_run -v`

Expected: FAIL because policy, log, and simulation modules do not exist.

- [ ] **Step 3: Implement policy, log dataclasses, serialization, and episode loop**

Use JSON-compatible lists at the serialization boundary and tuples in in-memory dataclasses. Store `lap_time_ms`, `finished`, `progress`, `damage`, `collision_count`, and `retire_reason` in `summary`. Keep the file schema independent of Box2D object types.

- [ ] **Step 4: Run focused, environment, and baseline tests**

Run: `D:\HAIC\haic-env\Scripts\python.exe -m unittest tests.test_local_simulator_run tests.test_local_simulator_environment tests.test_local_contract -v`

Expected: all tests pass and a two-step real-environment run completes without a GUI.

- [ ] **Step 5: Commit**

```text
git add local_simulator tests/test_local_simulator_run.py
git commit -m "feat: add local simulation logs"
```

### Task 4: Map and run command-line tools

**Files:**
- Create: `local_simulator/map.py`
- Create: `local_simulator/run.py`
- Create: `tests/test_local_simulator_cli.py`

**Interfaces:**
- `python -m local_simulator.map --track-id 1 --seed 42 --obstacle-mode official_plus_custom --output D:\HAIC\maps\map.json`
- `python -m local_simulator.run --map D:\HAIC\maps\map.json --output D:\HAIC\runs\run.json`
- `python -m local_simulator.run --map ... --record-frames`

The map command validates arguments, creates a deterministic map bundle, includes track preview points and official/custom obstacle preview data, and writes one `map.json`. The run command loads that file, uses `BaselinePolicy` in the first phase, and writes `run.json`. Output directories are created explicitly by the command, with the default root `D:\HAIC` and a project-local fallback when D: is unavailable.

- [ ] **Step 1: Write failing subprocess tests**

```python
class TestSimulatorCli(unittest.TestCase):
    def test_map_and_run_commands_create_artifacts(self):
        from local_simulator.map import main as map_main
        from local_simulator.run import main as run_main
        with tempfile.TemporaryDirectory() as root:
            map_path = Path(root) / "map.json"
            run_path = Path(root) / "run.json"
            map_main(["--track-id", "1", "--seed", "42", "--output", str(map_path)])
            run_main(["--map", str(map_path), "--output", str(run_path), "--max-steps", "1"])
            self.assertTrue(map_path.exists())
            self.assertTrue(run_path.exists())
            self.assertEqual(load_run_log(run_path).schema_version, 1)
```

- [ ] **Step 2: Run the focused test and verify it fails**

Run: `D:\HAIC\haic-env\Scripts\python.exe -m unittest tests.test_local_simulator_cli -v`

Expected: FAIL because the CLI modules and entry functions do not exist.

- [ ] **Step 3: Implement both CLI entry points**

Expose `main(argv: Sequence[str] | None = None) -> int` in each module so tests can call the command without spawning a new process. Preserve the same validation used by the schema module.

- [ ] **Step 4: Run CLI, simulator, and contract tests**

Run: `D:\HAIC\haic-env\Scripts\python.exe -m unittest tests.test_local_simulator_cli tests.test_local_simulator_run tests.test_local_contract -v`

Expected: all tests pass and the generated artifacts can be opened as JSON.

- [ ] **Step 5: Commit**

```text
git add local_simulator tests/test_local_simulator_cli.py
git commit -m "feat: add map and run CLIs"
```

### Task 5: Static site shell and map editor

**Files:**
- Create: `web_simulator/index.html`
- Create: `web_simulator/styles.css`
- Create: `web_simulator/app.js`
- Create: `tests/test_web_simulator_assets.py`

**Interfaces:**
- `web_simulator/index.html` loads only local `styles.css` and `app.js`.
- `app.js` exposes no framework global and initializes on `DOMContentLoaded`.
- File handlers accept `map.json` and `run.json` through `<input type="file">` and drag-and-drop.

Build the first screen with a map settings form, obstacle table, map canvas, log import area, playback controls, and metric cards. The map editor stores a client-side `mapSpec` object using the same field names as Python schema version 1. It can import a map, add/delete/edit custom obstacles, generate random normalized obstacle positions, and download the edited map as JSON. It never claims a map is official unless `obstacle_mode` says so.

- [ ] **Step 1: Write asset and markup tests**

```python
class TestWebAssets(unittest.TestCase):
    def test_site_has_local_assets_and_required_controls(self):
        html = Path("web_simulator/index.html").read_text(encoding="utf-8")
        self.assertIn('id="map-canvas"', html)
        self.assertIn('id="play-button"', html)
        self.assertIn('id="log-file"', html)
        self.assertIn('src="app.js"', html)
        self.assertNotIn("http://", html)
        self.assertNotIn("https://", html)
```

- [ ] **Step 2: Run the focused test and verify it fails**

Run: `D:\HAIC\haic-env\Scripts\python.exe -m unittest tests.test_web_simulator_assets -v`

Expected: FAIL because the static site does not exist.

- [ ] **Step 3: Implement HTML and CSS shell**

Use semantic controls with visible labels, a responsive two-column layout, a dark canvas panel, and status text for imported file errors. Keep the site dependency-free so it can be served by `python -m http.server` or ChatGPT Sites.

- [ ] **Step 4: Implement map import/edit/export behavior**

Parse JSON with `FileReader`, normalize and validate the same numeric ranges in the browser, render the imported track preview, and download a UTF-8 `map.json` without adding a server endpoint.

- [ ] **Step 5: Run asset tests and inspect the page**

Run: `D:\HAIC\haic-env\Scripts\python.exe -m unittest tests.test_web_simulator_assets -v`

Then serve it with `D:\HAIC\haic-env\Scripts\python.exe -m http.server 8000 --directory web_simulator` and verify the page opens without console-breaking missing assets.

- [ ] **Step 6: Commit**

```text
git add web_simulator tests/test_web_simulator_assets.py
git commit -m "feat: add map editor site shell"
```

### Task 6: Log replay, playback, and run comparison

**Files:**
- Modify: `web_simulator/index.html`
- Modify: `web_simulator/app.js`
- Modify: `web_simulator/styles.css`
- Modify: `tests/test_web_simulator_assets.py`

**Interfaces:**
- `loadRunLog(payload) -> RunState`
- `renderRunFrame(runState, stepIndex) -> void`
- `setPlaybackRunning(running: boolean) -> void`
- `advancePlayback() -> void`
- `summarizeRuns(runs) -> Array<RunSummary>`

`RunState` is a plain JavaScript object containing `{ log, stepIndex, running, speed }`; `RunSummary` contains the displayed run ID, finished flag, lap time, progress, damage, collision count, seed, and track ID.

The log viewer reads the embedded `track.points`, `steps`, `summary`, and optional `frames`. Normalize world coordinates to canvas bounds, draw track centerline and approximate road edges, draw obstacles, draw the vehicle orientation, and draw the traveled trajectory. If `frames` exists, show the selected frame in a camera panel; otherwise keep the minimap as the primary replay view.

The play loop uses `requestAnimationFrame`, advances at the selected wall-clock speed, and never sends a network request. It stops at the last step or when the run summary is terminal. Pause, one-step forward, one-step back, restart, and speed controls must not mutate the original log. Loading multiple logs adds rows to a comparison table ordered by finished first, then lower `lap_time_ms`, then higher progress.

- [ ] **Step 1: Extend the asset smoke tests with required playback controls**

```python
    def test_site_has_playback_and_metrics_controls(self):
        html = Path("web_simulator/index.html").read_text(encoding="utf-8")
        for element_id in ("pause-button", "step-forward", "step-back", "speed", "run-table"):
            self.assertIn(f'id="{element_id}"', html)
```

- [ ] **Step 2: Run the focused test and verify it fails**

Run: `D:\HAIC\haic-env\Scripts\python.exe -m unittest tests.test_web_simulator_assets -v`

Expected: FAIL because the playback controls are not present yet.

- [ ] **Step 3: Implement log parsing and canvas replay**

Reject schema versions other than `1`, show a readable error for missing `steps` or `track.points`, and keep all coordinate transforms in small functions so the map and trajectory use the same bounds.

- [ ] **Step 4: Implement automatic play and comparison table**

Use the last rendered step as the initial state, guard against duplicate animation loops, disable controls while no log is loaded, and display `FINISHED`/`DNF`, lap time, progress, damage, collisions, seed, and track ID.

- [ ] **Step 5: Run asset tests and manually replay a generated log**

Run: `D:\HAIC\haic-env\Scripts\python.exe -m unittest tests.test_web_simulator_assets -v`

Generate a small log with Task 4, open the site through the local static server, import the log, click play, pause, step, change speed, and verify the vehicle marker and metrics change without a backend request.

- [ ] **Step 6: Commit**

```text
git add web_simulator tests/test_web_simulator_assets.py
git commit -m "feat: add log replay and run comparison"
```

### Task 7: Integration verification, documentation, and handoff

**Files:**
- Modify: `README.md`
- Create: `tests/test_local_simulator_integration.py`
- Create: `examples/demo_map.json` only if it remains below 20 KiB

Add README instructions for Python 3.11, the D-drive artifact directories, map generation, baseline run, optional frame recording, and serving the static site:

```text
D:\HAIC\haic-env\Scripts\python.exe -m local_simulator.map --track-id 1 --seed 42 --output D:\HAIC\maps\map.json
D:\HAIC\haic-env\Scripts\python.exe -m local_simulator.run --map D:\HAIC\maps\map.json --output D:\HAIC\runs\run.json
D:\HAIC\haic-env\Scripts\python.exe -m http.server 8000 --directory web_simulator
```

- [ ] **Step 1: Write the end-to-end integration test**

```python
class TestLocalSimulatorIntegration(unittest.TestCase):
    def test_map_run_log_contains_replay_data(self):
        spec = MapSpec(1, 42, "official_plus_custom", (CustomObstacle(0.5, 0.0, 1.2),), 2, 4)
        map_bundle = build_map_bundle(spec)
        run_log = run_episode(spec, BaselinePolicy(), record_frames=False)
        self.assertGreater(len(map_bundle.track.points), 0)
        self.assertEqual(len(run_log.steps), 2)
        self.assertIn("lap_time_ms", run_log.summary)
```

- [ ] **Step 2: Run the full test suite and verify integration failure if any**

Run: `D:\HAIC\haic-env\Scripts\python.exe -m unittest discover -s tests -v`

Expected after all prior tasks: existing contract tests pass, the server parity test is skipped when the server repository is absent, and all new simulator/site asset tests pass.

- [ ] **Step 3: Add README and small-file checks**

Document that the site replays uploaded logs and does not execute official physics in the browser. Do not commit generated runs or frame bundles. Confirm `examples/demo_map.json` is small before adding it.

- [ ] **Step 4: Run final verification**

Run:

```text
D:\HAIC\haic-env\Scripts\python.exe -m unittest discover -s tests -v
D:\HAIC\haic-env\Scripts\python.exe -m local_simulator.map --track-id 1 --seed 42 --output D:\HAIC\maps\verification-map.json
D:\HAIC\haic-env\Scripts\python.exe -m local_simulator.run --map D:\HAIC\maps\verification-map.json --output D:\HAIC\runs\verification-run.json --max-steps 10
git diff --check
git status --short
```

Then perform the browser smoke test from Task 6 with the generated verification log.

- [ ] **Step 5: Commit**

```text
git add README.md tests/test_local_simulator_integration.py examples
git commit -m "docs: document local simulation and log replay"
```

## Handoff to agent phase

After Task 7 is green, add the agent phase as a separate plan. It should implement a policy adapter that consumes the official `(4, 84, 84)` observation, run it through the same `run_episode` loop, and produce the same `run.json` schema so the website needs no changes.

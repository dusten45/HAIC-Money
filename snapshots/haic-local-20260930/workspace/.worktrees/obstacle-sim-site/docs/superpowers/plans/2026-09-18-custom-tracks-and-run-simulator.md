# 자체 트랙 및 실행 로그 생성기 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** seed 기반 공식 맵과 독립적인 자체 트랙을 만들고, 공식·자체 맵을 로컬에서 실제 주행해 run.json으로 저장하며, 웹에서 생성·실행·재생·비교할 수 있게 한다.

**Architecture:** 기존 공식 CarRacing 환경과 파일은 그대로 두고, schema 2 맵 계약과 CustomTrackEnvironment를 별도 백엔드로 추가한다. 공통 Policy와 SimulationSession이 baseline, agent.py, 브라우저 수동 주행을 동일한 로그 형식으로 기록한다. 정적 웹사이트는 파일 보기 전용을 유지하면서 로컬 통합 서버에서는 같은 origin의 API로 실제 실행을 제어하고, 원격 서비스는 맵과 로그만 저장한다.

**Tech Stack:** Python 3.11, Gymnasium 0.29.1, vendored Box2D/pygame, NumPy 1.26, OpenCV 4.8, Python standard-library http.server, 정적 HTML/CSS/JavaScript, unittest.

**Spec:** docs/superpowers/specs/2026-09-18-custom-tracks-and-run-simulator-design.md

## Global Constraints

- 공식 core/, env_wrapper.py, damage.py는 수정하지 않는다.
- 공식 맵은 기존 track_id·seed와 공식 CarRacing backend를 계속 사용한다.
- 자체 맵은 custom-track-* 문자열 map_id와 중심선 geometry를 사용한다.
- schema 1 공식 맵과 schema 1 실행 로그를 읽을 수 있어야 하며 새로 저장하는 파일은 schema 2를 사용한다.
- 기본 산출물은 D:/HAIC/maps와 D:/HAIC/runs에 저장하고 경로가 없을 때만 프로젝트 내부 fallback을 사용한다.
- 서버에서는 Agent 코드와 Box2D 주행을 실행하지 않는다.
- 새 Python dependency를 추가하지 않고 현재 설치된 환경을 사용한다.
- 모든 새 입력은 유한성, 범위, 개수, 파일 크기를 검증한다.
- 브라우저는 Box2D를 재구현하지 않고 로그를 재생한다.
- 각 task는 해당 테스트가 통과한 뒤 독립적인 git commit으로 저장한다.

## File Map

### Existing files to modify

- local_simulator/schema.py: 공식 schema 1 호환과 공식·자체 맵 schema 2 dispatch.
- local_simulator/environment.py: map kind에 따른 backend 선택과 preview bundle dispatch.
- local_simulator/simulation.py: 공통 session/recorder를 사용하는 episode 실행.
- local_simulator/simulation_types.py: schema 2 run metadata와 custom track snapshot.
- local_simulator/logging.py: schema 1 reader와 schema 2 writer.
- local_simulator/policies.py: AgentPolicy와 ManualPolicy.
- local_simulator/map.py: 공식·자체 맵 CLI와 D 드라이브 저장.
- local_simulator/run.py: 정책 선택과 schema 2 log 출력.
- web_simulator/index.html: 자체 맵 제작·실행·수동 조작 UI.
- web_simulator/styles.css: 새 제작기와 실행 패널 스타일.
- web_simulator/app.js: custom map/editor, local API, manual action, auto-load.
- README.md: 로컬 서버·자체 트랙·run.json 사용법.
- tests/test_local_simulator_schema.py: schema 2 공식/자체 round-trip과 legacy fixture.
- tests/test_local_simulator_environment.py: backend dispatch와 custom environment 계약.
- tests/test_local_simulator_run.py: schema 2 log와 legacy load 검증.
- tests/test_local_simulator_cli.py: custom map/run CLI 검증.
- tests/test_local_simulator_integration.py: official regression과 custom integration.
- tests/test_web_simulator_assets.py: 새 DOM/API 계약의 정적 검사.

### New files to create

- local_simulator/track_model.py: CustomTrackGeometry, CustomMapSpec, fingerprint와 geometry validation.
- local_simulator/track_generator.py: deterministic templates와 batch generation.
- local_simulator/custom_environment.py: 공식 CarRacing 동역학을 재사용하는 custom track backend.
- local_simulator/session.py: 한 번의 실행을 시작·step·finish하는 상태ful runner.
- local_simulator/api.py: local HTTP JSON API와 session registry.
- local_simulator/web.py: static site와 local API를 same-origin으로 제공하는 launcher.
- haic_service/__init__.py: 원격 map/log service package.
- haic_service/storage.py: JSON artifact 저장과 path/size 제한.
- haic_service/app.py: 원격 health/map/run-upload API.
- tests/test_track_model.py: geometry validation과 fingerprint 테스트.
- tests/test_track_generator.py: template deterministic/batch 테스트.
- tests/test_custom_environment.py: custom Box2D 환경 smoke 테스트.
- tests/test_local_simulator_api.py: local HTTP API integration 테스트.
- tests/test_haic_service.py: 원격 storage/API 보안·계약 테스트.
- docs/remote-deployment.md: 서버 venv·systemd·Caddy 연결 절차.

---

### Task 1: Versioned map contract and legacy compatibility

**Files:**
- Create: local_simulator/track_model.py
- Modify: local_simulator/schema.py
- Modify: local_simulator/__init__.py
- Modify: tests/test_local_simulator_schema.py
- Create: tests/test_track_model.py

**Interfaces:**
- CustomTrackGeometry(centerline, width, start_index=0, direction=1)
- CustomMapSpec(map_id, geometry, obstacles, max_steps, frame_skip, generator, metadata)
- MapDocument = MapSpec | CustomMapSpec
- map_from_dict(payload) -> MapDocument
- map_to_dict(document) -> dict
- MAP_SCHEMA_VERSION = 2 and LEGACY_SCHEMA_VERSION = 1

- [ ] Step 1: Write failing contract tests.

~~~python
def test_custom_map_round_trip_preserves_geometry(self):
    original = CustomMapSpec(
        map_id="custom-track-0001",
        geometry=CustomTrackGeometry(
            centerline=((0.0, 0.0), (12.0, 0.0), (12.0, 12.0), (0.0, 12.0)),
            width=8.0,
        ),
        obstacles=(),
        max_steps=2000,
        frame_skip=4,
        generator=(("template", "oval"), ("design_seed", 7)),
    )
    self.assertEqual(map_from_dict(map_to_dict(original)), original)

def test_schema_one_payload_is_read_as_official_map(self):
    result = map_from_dict({"track_id": 1, "seed": 42})
    self.assertEqual(result.track_id, 1)
    self.assertEqual(result.seed, 42)
    self.assertEqual(result.map_kind, "official")
~~~

- [ ] Step 2: Run the focused tests and verify they fail.

~~~powershell
D:/HAIC/haic-env/Scripts/python.exe -m unittest tests.test_local_simulator_schema tests.test_track_model -v
~~~

Expected: FAIL because the custom dataclasses and schema dispatch do not exist.

- [ ] Step 3: Implement the minimal contract.

Keep MapSpec constructible by all existing positional callers. Add CustomMapSpec and dispatch based on map_kind. Treat payloads without schema_version as legacy official schema 1. Validate custom map IDs with the custom-track- prefix, finite centerline coordinates, positive finite width, valid start index/direction, custom obstacle limits, max_steps, and frame_skip. A custom map uses custom_only obstacle mode. map_to_dict writes new official maps with map_kind official and custom maps with map_kind custom.

- [ ] Step 4: Run focused and existing schema tests.

~~~powershell
D:/HAIC/haic-env/Scripts/python.exe -m unittest tests.test_local_simulator_schema tests.test_track_model -v
~~~

Expected: PASS, including old MapSpec tests and new custom round-trip tests.

- [ ] Step 5: Commit.

~~~powershell
git add local_simulator/schema.py local_simulator/track_model.py local_simulator/__init__.py tests/test_local_simulator_schema.py tests/test_track_model.py
git commit -m "feat: add versioned official and custom map schema"
~~~

### Task 2: Deterministic custom track generator and geometry validation

**Files:**
- Create: local_simulator/track_generator.py
- Modify: local_simulator/track_model.py
- Create: tests/test_track_generator.py
- Modify: tests/test_track_model.py

**Interfaces:**
- validate_custom_geometry(geometry: CustomTrackGeometry) -> None
- generate_custom_map(map_id, design_seed, template, width=8.0, max_steps=2000, frame_skip=4) -> CustomMapSpec
- generate_custom_maps(start_seed, count, template, prefix="custom-track") -> tuple[CustomMapSpec, ...]
- custom_geometry_fingerprint(geometry) -> str
- Supported templates: oval, s_curve, hairpin, chicane.

- [ ] Step 1: Write failing deterministic and validation tests.

~~~python
def test_same_design_seed_has_same_geometry_and_fingerprint(self):
    first = generate_custom_map("custom-track-0001", 90421, "hairpin")
    second = generate_custom_map("custom-track-0001", 90421, "hairpin")
    self.assertEqual(first, second)

def test_different_design_seed_changes_geometry(self):
    first = generate_custom_map("custom-track-0001", 1, "oval")
    second = generate_custom_map("custom-track-0002", 2, "oval")
    self.assertNotEqual(first.geometry.centerline, second.geometry.centerline)

def test_self_intersection_is_rejected(self):
    geometry = CustomTrackGeometry(
        centerline=((0, 0), (10, 10), (0, 10), (10, 0)), width=8
    )
    with self.assertRaisesRegex(ValueError, "self-intersection"):
        validate_custom_geometry(geometry)
~~~

- [ ] Step 2: Run generator tests and verify they fail.

~~~powershell
D:/HAIC/haic-env/Scripts/python.exe -m unittest tests.test_track_generator tests.test_track_model -v
~~~

Expected: FAIL because the generator and validator are missing.

- [ ] Step 3: Implement deterministic templates and validator.

Use random.Random(design_seed) only. Generate a sampled closed centerline in world units. Require at least 12 distinct points, finite coordinates, positive width, valid start index/direction, nonzero segments, no non-adjacent segment intersections, and enough turn clearance for the selected width. Store generator fields as sorted key/value pairs so serialization is stable.

~~~python
def generate_custom_map(map_id, design_seed, template, width=8.0,
                        max_steps=2000, frame_skip=4):
    geometry = _generate_geometry(template, design_seed, width)
    validate_custom_geometry(geometry)
    return CustomMapSpec(
        map_id=map_id,
        geometry=geometry,
        obstacles=(),
        max_steps=max_steps,
        frame_skip=frame_skip,
        generator=(("design_seed", design_seed), ("template", template)),
    )
~~~

- [ ] Step 4: Run generator tests and schema tests.

~~~powershell
D:/HAIC/haic-env/Scripts/python.exe -m unittest tests.test_track_generator tests.test_track_model tests.test_local_simulator_schema -v
~~~

Expected: PASS; all four templates are deterministic and invalid geometry produces a specific error.

- [ ] Step 5: Commit.

~~~powershell
git add local_simulator/track_model.py local_simulator/track_generator.py tests/test_track_model.py tests/test_track_generator.py
git commit -m "feat: generate deterministic custom track geometry"
~~~

### Task 3: Map CLI and preview artifacts for official and custom maps

**Files:**
- Modify: local_simulator/map.py
- Modify: local_simulator/environment.py
- Create: local_simulator/preview.py
- Modify: tests/test_local_simulator_cli.py
- Modify: tests/test_local_simulator_environment.py

**Interfaces:**
- build_map_preview(document: MapDocument) -> dict
- local_simulator.map.main accepts --kind official|custom.
- Custom CLI accepts --map-id, --design-seed, --template, --width, and repeated --control-point X,Y.
- Existing official CLI invocation remains valid.

- [ ] Step 1: Write failing CLI tests.

~~~python
def test_custom_map_command_creates_geometry_preview(self):
    map_main([
        "--kind", "custom",
        "--map-id", "custom-track-0001",
        "--design-seed", "90421",
        "--template", "hairpin",
        "--output", str(map_path),
    ])
    payload = json.loads(map_path.read_text(encoding="utf-8"))
    self.assertEqual(payload["map_kind"], "custom")
    self.assertEqual(payload["map_id"], "custom-track-0001")
    self.assertGreaterEqual(len(payload["geometry"]["centerline"]), 12)
    self.assertIn("track", payload["preview"])
~~~

Add a second test proving two design seeds write different centerlines and retain the same schema when reloaded.

- [ ] Step 2: Run the focused CLI tests and verify they fail.

~~~powershell
D:/HAIC/haic-env/Scripts/python.exe -m unittest tests.test_local_simulator_cli -v
~~~

Expected: FAIL because the CLI only accepts official track_id/seed inputs.

- [ ] Step 3: Implement preview and CLI dispatch.

Keep the current official path through build_map_bundle and add a pure-data custom path. For a custom map, project centerline points into TrackSnapshot entries of (progress, tangent_angle, x, y), and place preview obstacles from the custom geometry. Do not instantiate Box2D merely to export a custom map. Preserve the D:/HAIC artifact-root fallback.

~~~python
def build_map_preview(document):
    if getattr(document, "map_kind", "official") == "custom":
        return _custom_preview(document)
    return _official_preview(document)
~~~

- [ ] Step 4: Run official regression and custom CLI tests.

~~~powershell
D:/HAIC/haic-env/Scripts/python.exe -m unittest tests.test_local_simulator_cli tests.test_local_simulator_environment -v
~~~

Expected: PASS for existing official preview/obstacle tests and new custom artifact tests.

- [ ] Step 5: Commit.

~~~powershell
git add local_simulator/map.py local_simulator/environment.py local_simulator/preview.py tests/test_local_simulator_cli.py tests/test_local_simulator_environment.py
git commit -m "feat: add custom track map generation artifacts"
~~~

### Task 4: Custom Box2D environment with the common observation/action contract

**Files:**
- Create: local_simulator/custom_environment.py
- Modify: local_simulator/environment.py
- Create: tests/test_custom_environment.py
- Modify: tests/test_local_simulator_integration.py

**Interfaces:**
- CustomCarRacing(CarRacing) overrides custom track construction and custom finish-line geometry while reusing vendored car, contact listener, rendering, and obstacle bodies.
- create_environment(document, render_mode="rgb_array") -> tuple[CarEnvironment, CarRacing]
- reset_environment(environment, document) -> tuple[np.ndarray, dict]
- snapshot_track(environment) -> TrackSnapshot
- map_custom_obstacles(environment, document)

- [ ] Step 1: Write failing custom environment smoke tests.

~~~python
def test_custom_environment_has_track_and_expected_observation(self):
    document = generate_custom_map("custom-track-0001", 90421, "hairpin")
    environment, raw = create_environment(document, render_mode=None)
    try:
        observation, _ = reset_environment(environment, document)
        self.assertEqual(observation.shape, (4, 84, 84))
        self.assertEqual(len(raw.track), len(document.geometry.centerline))
        next_observation, *_ = environment.step(np.array([0.0, 1.0, 0.0]))
        self.assertEqual(next_observation.shape, (4, 84, 84))
    finally:
        environment.close()
~~~

Also test deterministic custom snapshots and that one custom obstacle appears in raw.obstacles after reset.

- [ ] Step 2: Run custom environment tests and verify they fail.

~~~powershell
D:/HAIC/haic-env/Scripts/python.exe -m unittest tests.test_custom_environment tests.test_local_simulator_integration -v
~~~

Expected: FAIL because create_environment only constructs official CarRacing.

- [ ] Step 3: Implement custom track construction.

Implement CustomCarRacing._create_track by creating one four-vertex road tile per centerline point using the current point and its previous point, marking the fixture as a sensor with road_friction=1.0, assigning sequential idx values, and adding polygons to road_poly for rendering. This keeps len(self.track) equal to the stored centerline length. Use geometry width for road boundaries and the finish tracker. Call the inherited reset lifecycle so contact listeners, Car, obstacle contact tracking, and rendering stay compatible.

~~~python
class CustomCarRacing(CarRacing):
    def __init__(self, geometry, render_mode=None):
        super().__init__(render_mode=render_mode, continuous=True)
        self.custom_geometry = geometry

    def _create_track(self):
        # Build self.track, self.road, and self.road_poly from geometry.
        return True

    def reset(self, *, seed=None, options=None):
        observation, info = super().reset(seed=seed, options={})
        self._replace_finish_line_for_custom_width()
        return observation, info
~~~

The custom backend must not call build_track_variables or add official obstacles. Reuse CarEnvironment for frame stacking, damage, raw-frame skipping, off-track detection, and info fields. Update environment.py only for map-kind dispatch and geometry-aware custom obstacle projection.

- [ ] Step 4: Run custom and official environment tests.

~~~powershell
D:/HAIC/haic-env/Scripts/python.exe -m unittest tests.test_custom_environment tests.test_local_simulator_environment tests.test_local_simulator_integration -v
~~~

Expected: PASS, including six official obstacles for official maps and zero official obstacles for custom maps.

- [ ] Step 5: Commit.

~~~powershell
git add local_simulator/custom_environment.py local_simulator/environment.py tests/test_custom_environment.py tests/test_local_simulator_integration.py
git commit -m "feat: run custom maps in a separate Box2D backend"
~~~

### Task 5: Unified policy session, Agent adapter, and schema 2 run logs

**Files:**
- Create: local_simulator/session.py
- Modify: local_simulator/policies.py
- Modify: local_simulator/simulation.py
- Modify: local_simulator/simulation_types.py
- Modify: local_simulator/logging.py
- Modify: local_simulator/run.py
- Modify: tests/test_local_simulator_run.py
- Modify: tests/test_local_simulator_integration.py

**Interfaces:**
- AgentPolicy(agent_path: Path, project_root: Path) with reset, act, and name="agent".
- ManualPolicy with set_action, reset, act.
- SimulationSession.start(document, policy, record_frames=False), step(action=None) -> RunStep, finish(reason=None) -> RunLog, and closed.
- run_episode(document, policy, record_frames=False) -> RunLog remains the batch API.
- run_log_from_dict accepts schema 1 and schema 2; save_run_log always writes schema 2.

- [ ] Step 1: Write failing tests for policy modes and log versioning.

~~~python
def test_custom_baseline_run_writes_map_reference(self):
    document = generate_custom_map("custom-track-0001", 90421, "oval")
    result = run_episode(document, BaselinePolicy())
    self.assertEqual(result.schema_version, 2)
    self.assertEqual(result.run["map_ref"]["map_id"], "custom-track-0001")
    self.assertEqual(result.run["environment"], "custom-v1")

def test_legacy_schema_one_log_still_loads(self):
    loaded = run_log_from_dict(legacy_payload)
    self.assertEqual(len(loaded.steps), 1)
    self.assertEqual(loaded.run["map"]["track_id"], 1)

def test_manual_session_records_explicit_action(self):
    session = SimulationSession.start(document, ManualPolicy(), record_frames=False)
    step = session.step((0.25, 0.5, 0.0))
    self.assertEqual(step.action, (0.25, 0.5, 0.0))
~~~

- [ ] Step 2: Run run/log tests and verify they fail.

~~~powershell
D:/HAIC/haic-env/Scripts/python.exe -m unittest tests.test_local_simulator_run tests.test_local_simulator_integration -v
~~~

Expected: FAIL because logs use schema 1 and there is no stateful manual session or custom map dispatch.

- [ ] Step 3: Implement policy adapters and the stateful session.

AgentPolicy loads the explicit workspace agent.py path and does not accept arbitrary HTTP import paths. Instantiate Agent, call its optional reset, and pass the (4, 84, 84) observation to act. ManualPolicy stores the latest validated action and defaults to [0, 0, 0].

Move the per-step recording from run_episode into SimulationSession.step. The session owns environment, policy, frame encoder, start time, last info, and steps. finish closes the environment exactly once and builds the existing summary fields.

Use separate map/run version constants. Preserve old run.map data while adding run.map_ref, run.environment, and structured run.policy to new logs.

- [ ] Step 4: Run unit, integration, and CLI run tests.

~~~powershell
D:/HAIC/haic-env/Scripts/python.exe -m unittest tests.test_local_simulator_run tests.test_local_simulator_integration tests.test_local_simulator_cli -v
~~~

Expected: PASS for official legacy fixtures, schema 2 logs, custom baseline logs, and existing CLI artifacts.

- [ ] Step 5: Commit.

~~~powershell
git add local_simulator/session.py local_simulator/policies.py local_simulator/simulation.py local_simulator/simulation_types.py local_simulator/logging.py local_simulator/run.py tests/test_local_simulator_run.py tests/test_local_simulator_integration.py
git commit -m "feat: generate versioned run logs for agent and manual policies"
~~~

### Task 6: Local same-origin API and simulator launcher

**Files:**
- Create: local_simulator/api.py
- Create: local_simulator/web.py
- Create: tests/test_local_simulator_api.py
- Modify: local_simulator/__init__.py
- Modify: README.md

**Interfaces:**
- SimulationRegistry.create(document, policy_kind, record_frames) -> str.
- SimulationRegistry.action(run_id, action) -> dict.
- SimulationRegistry.finish(run_id) -> tuple[RunLog, Path].
- LocalApiHandler endpoints: GET /api/health, POST /api/maps/generate, POST /api/maps/save, POST /api/runs/agent, POST /api/runs/start, POST /api/runs/{run_id}/action, POST /api/runs/{run_id}/finish, GET /api/runs/{run_id}.
- python -m local_simulator.web --host 127.0.0.1 --port 8765 --project-root . serves web_simulator and the API from one origin.

- [ ] Step 1: Write failing HTTP contract tests.

~~~python
def test_health_and_custom_map_generation(self):
    response = self.request("GET", "/api/health")
    self.assertEqual(response["status"], "ok")
    response = self.request(
        "POST",
        "/api/maps/generate",
        {"map_kind": "custom", "map_id": "custom-track-api",
         "design_seed": 7, "template": "oval"},
    )
    self.assertEqual(response["map_kind"], "custom")
    self.assertGreaterEqual(len(response["geometry"]["centerline"]), 12)

def test_manual_run_start_action_finish_writes_run(self):
    started = self.request("POST", "/api/runs/start",
                           {"map": custom_payload, "policy": "manual"})
    stepped = self.request(
        "POST", "/api/runs/%s/action" % started["run_id"],
        {"action": [0, 1, 0]},
    )
    self.assertEqual(stepped["step"]["action"], [0.0, 1.0, 0.0])
    finished = self.request(
        "POST", "/api/runs/%s/finish" % started["run_id"], {})
    self.assertTrue(Path(finished["output"]).exists())
~~~

- [ ] Step 2: Run API tests and verify they fail.

~~~powershell
D:/HAIC/haic-env/Scripts/python.exe -m unittest tests.test_local_simulator_api -v
~~~

Expected: FAIL because no local HTTP launcher or session registry exists.

- [ ] Step 3: Implement registry and standard-library HTTP server.

Use ThreadingHTTPServer and a lock-protected dictionary keyed by secrets.token_urlsafe(18). Reject unknown or finished IDs with HTTP 404/409. Save completed logs under the configured artifact root and return the resolved path as display data, never as an input path.

Use the fixed workspace root passed to web.py for Agent mode. The agent endpoint loads that root's agent.py and never accepts an arbitrary import path from the browser. Limit map/session request bodies to 256 KiB, reject malformed JSON with HTTP 400, and return {"error": "..."} consistently.

Serve only files under web_simulator using a normalized relative path. Keep plain python -m http.server documented for view-only use.

- [ ] Step 4: Run API tests and launcher smoke test.

~~~powershell
D:/HAIC/haic-env/Scripts/python.exe -m unittest tests.test_local_simulator_api -v
D:/HAIC/haic-env/Scripts/python.exe -m local_simulator.web --help
~~~

Expected: PASS and the launcher prints its bind address without importing Box2D until a run is requested.

- [ ] Step 5: Commit.

~~~powershell
git add local_simulator/api.py local_simulator/web.py local_simulator/__init__.py tests/test_local_simulator_api.py README.md
git commit -m "feat: add local run simulator API and launcher"
~~~

### Task 7: Web custom-track editor, run controls, and automatic replay

**Files:**
- Modify: web_simulator/index.html
- Modify: web_simulator/styles.css
- Modify: web_simulator/app.js
- Modify: tests/test_web_simulator_assets.py

**Interfaces:**
- Existing window.HAICSimulator replay API remains available.
- Add client functions generateCustomMap(), applyCustomMap(payload), startRun(policy), sendManualAction(action), and finishRun().
- Use same-origin /api/* when available; if /api/health fails, keep file import/replay active and show file-only status.
- Keyboard controls: ArrowLeft/ArrowRight steering, ArrowUp gas, Space brake; key release returns that control to zero.

- [ ] Step 1: Write failing static asset tests.

~~~python
def test_site_contains_custom_generator_and_run_controls(self):
    html = Path("web_simulator/index.html").read_text(encoding="utf-8")
    for element_id in (
        "map-kind", "custom-template", "design-seed", "generate-custom-map",
        "start-agent-run", "start-manual-run", "finish-manual-run",
        "local-api-status",
    ):
        self.assertIn('id="%s"' % element_id, html)

def test_script_contains_api_fallback_and_manual_action_names(self):
    script = Path("web_simulator/app.js").read_text(encoding="utf-8")
    for token in (
        "/api/health", "generateCustomMap", "startRun",
        "sendManualAction", "custom-track",
    ):
        self.assertIn(token, script)
~~~

- [ ] Step 2: Run browser asset tests and verify they fail.

~~~powershell
D:/HAIC/haic-env/Scripts/python.exe -m unittest tests.test_web_simulator_assets -v
~~~

Expected: FAIL because the current page only edits official seed/track fields and imports existing logs.

- [ ] Step 3: Implement the UI flow.

Add a map-kind switch. Official mode preserves current fields. Custom mode shows template, design seed, width, map ID, generate, control-point editing, validation messages, and the existing obstacle table. Render custom centerline/boundaries using the existing world-coordinate projection and keep official/custom obstacle legends distinct.

Add an execution panel with map selector, policy selector, frame recording checkbox, start, pause, step, manual keyboard status, and finish/save. startRun("agent") uses the local API for a full Agent run; startRun("manual") opens a session and sends the current action at a bounded interval. On finish, parse the returned log, add it to comparison, and call the existing playback renderer.

Use AbortController for API requests, render errors in the existing status area, and stop the manual timer on pause, finish, page visibility loss, or error. Do not put Python paths or secrets in the page.

- [ ] Step 4: Run static tests and browser verification.

~~~powershell
node --check web_simulator/app.js
D:/HAIC/haic-env/Scripts/python.exe -m unittest tests.test_web_simulator_assets -v
~~~

Start the integrated launcher:

~~~powershell
D:/HAIC/haic-env/Scripts/python.exe -m local_simulator.web --host 127.0.0.1 --port 8765 --project-root .
~~~

Expected: custom map creation, full road rendering, Agent/baseline run completion, manual keyboard action, automatic run.json replay, pause/resume/step, and file-only fallback with the plain static server.

- [ ] Step 5: Commit.

~~~powershell
git add web_simulator/index.html web_simulator/styles.css web_simulator/app.js tests/test_web_simulator_assets.py
git commit -m "feat: connect custom maps and run generation to the web app"
~~~

### Task 8: Remote map and run-log service without server-side simulation

**Files:**
- Create: haic_service/__init__.py
- Create: haic_service/storage.py
- Create: haic_service/app.py
- Create: tests/test_haic_service.py
- Create: docs/remote-deployment.md
- Modify: README.md

**Interfaces:**
- ArtifactStore(root, max_run_bytes=50 * 1024 * 1024) with save_map, list_maps, load_map, and save_run.
- HAICServiceHandler endpoints: GET /api/health, GET /api/maps, GET /api/maps/{map_id}, POST /api/maps/generate, POST /api/runs/upload.
- python -m haic_service.app --host 127.0.0.1 --port 8125 binds the service locally.
- HAIC_DATA_ROOT, HAIC_API_TOKEN, and HAIC_ALLOWED_ORIGIN are read from environment variables.

- [ ] Step 1: Write failing service/storage tests.

~~~python
def test_store_rejects_path_traversal(self):
    with self.assertRaises(ValueError):
        self.store.save_map({"map_id": "../outside", "map_kind": "custom"})

def test_write_endpoint_requires_token(self):
    response = self.request("POST", "/api/maps/generate", payload, headers={})
    self.assertEqual(response.status, 401)

def test_run_upload_rejects_oversized_body(self):
    response = self.request(
        "POST", "/api/runs/upload",
        b"x" * (50 * 1024 * 1024 + 1),
    )
    self.assertEqual(response.status, 413)
~~~

Also test deterministic custom map generation without importing or creating a Box2D environment.

- [ ] Step 2: Run service tests and verify they fail.

~~~powershell
D:/HAIC/haic-env/Scripts/python.exe -m unittest tests.test_haic_service -v
~~~

Expected: FAIL because the remote package and artifact store are absent.

- [ ] Step 3: Implement JSON-file storage and API.

Store maps as root/maps/safe-map-id.json and runs as root/runs/safe-run-id.json. Accept only safe IDs matching [A-Za-z0-9][A-Za-z0-9_-]{0,63} after the required map/run prefix. Validate map payloads with map_from_dict and run payloads with run_log_from_dict before writing. Reject map request bodies over 256 KiB and run uploads over 50 MiB. Apply X-HAIC-Token to write routes when HAIC_API_TOKEN is configured, and emit one configured CORS origin.

The remote generator may call pure generate_custom_map but must not call create_environment, run_episode, or load agent.py. The service is storage/API only.

- [ ] Step 4: Run service tests and local deployment smoke test.

~~~powershell
D:/HAIC/haic-env/Scripts/python.exe -m unittest tests.test_haic_service -v
D:/HAIC/haic-env/Scripts/python.exe -m haic_service.app --help
~~~

Expected: PASS, with deterministic maps, rejected unsafe paths/oversized uploads, and token enforcement.

- [ ] Step 5: Commit.

~~~powershell
git add haic_service tests/test_haic_service.py docs/remote-deployment.md README.md
git commit -m "feat: add remote map and run-log storage service"
~~~

### Task 9: Documentation, full verification, and deployment handoff

**Files:**
- Modify: README.md
- Modify: docs/remote-deployment.md
- Modify: tests/test_server_parity.py only if an explicit custom-backend exclusion is needed.
- Do not commit generated D:/HAIC maps or runs.

- [ ] Step 1: Add runnable documentation.

Document these exact commands:

~~~powershell
D:/HAIC/haic-env/Scripts/python.exe -m local_simulator.web --host 127.0.0.1 --port 8765 --project-root .
D:/HAIC/haic-env/Scripts/python.exe -m local_simulator.map --kind custom --map-id custom-track-0001 --design-seed 90421 --template hairpin --output D:/HAIC/maps/custom-track-0001.json
D:/HAIC/haic-env/Scripts/python.exe -m local_simulator.run --map D:/HAIC/maps/custom-track-0001.json --output D:/HAIC/runs/custom-track-0001-run.json
~~~

Explain the difference between official evaluation maps and custom validation maps, and explain that port 3125 is SSH rather than the HTTP application port.

- [ ] Step 2: Run the complete verification suite.

~~~powershell
node --check web_simulator/app.js
git diff --check
D:/HAIC/haic-env/Scripts/python.exe -W ignore::DeprecationWarning -m unittest discover -s tests -v
~~~

Expected: all applicable tests pass, the known server parity test is skipped when the server repository is absent, and no syntax or whitespace errors are reported.

- [ ] Step 3: Exercise end-to-end artifacts.

Run the custom CLI and integrated launcher, create one baseline/Agent log and one manual log, then confirm:

~~~powershell
Get-ChildItem D:/HAIC/maps/custom-track-0001.json
Get-ChildItem D:/HAIC/runs/*.json
~~~

Open both logs in the browser and confirm map identity, environment badge, progress, damage, collision count, replay controls, and comparison table. Inspect the browser console for zero uncaught errors.

- [ ] Step 4: Review changed-file boundary.

~~~powershell
git status --short
git diff --name-only main...HEAD
~~~

Expected: changes are limited to simulator, web UI, tests, documentation, and haic_service. core/, env_wrapper.py, damage.py, and submission files remain untouched.

- [ ] Step 5: Commit final documentation and verification adjustments.

~~~powershell
git add README.md docs/remote-deployment.md tests/test_server_parity.py
git commit -m "docs: document custom track and run simulator workflow"
~~~

The Caddy configuration and remote service installation are performed only after the final HTTP hostname is supplied. The implementation handoff reports the local URL, artifact locations, tests passed, and whether remote routing is waiting on that hostname.

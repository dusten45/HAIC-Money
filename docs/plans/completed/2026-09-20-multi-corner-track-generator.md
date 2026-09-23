> **Completed historical implementation plan, relocated 2026-09-23.** Its unchecked
> checklist is preserved as the original planning record, not a current execution
> queue. Completion here means the implementation work is no longer active; use
> `docs/architecture/overview.md` and code/tests for current behavior.

# Multi-Corner Custom Track Generator Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Generate deterministic custom circuits whose corner count, sequence, radius, and straight lengths make layouts visibly distinct for agent generalization testing.

**Architecture:** Keep map schema 2 and the existing generation API. Add a versioned Python generator that composes seeded corner profiles into a smooth closed centerline, extend geometry safety checks, then port the same PRNG and construction rules to the browser-only fallback. Store the recipe summary in generator metadata and show the corner summary in the map builder.

**Tech Stack:** Python standard library, existing Box2D custom environment, browser JavaScript, Node.js client tests, Python `unittest`.

**Spec:** `../../architecture/history/2026-09-20-multi-corner-track-generator-design.md`

## Global Constraints

- Preserve `MAP_SCHEMA_VERSION = 2` and the current `/api/maps/generate` request contract.
- Preserve existing template IDs `oval`, `s_curve`, `hairpin`, and `chicane`; add `technical`.
- Seeds remain unsigned 32-bit integers in `0..4294967295`; generated coordinates remain finite and within the existing `±100000` model limit.
- Use a versioned, explicit 32-bit PRNG shared by Python and JavaScript; do not change the obstacle PRNG.
- Keep official track and physics files (`core/`, `env_wrapper.py`, `damage.py`) unchanged.
- Keep obstacles independent from road generation; generated custom tracks have no obstacles unless the caller explicitly supplies them.
- Generated maps and run logs remain local artifacts; tests use temporary directories rather than writing to `D:\HAIC`.
- Do not disable configured SSH commit signing. If signing fails, stop and ask before attempting an unsigned commit.
- Before Task 1, preserve and resolve the pre-existing staged Task 7 web change in its own normally SSH-signed commit. After this plan is approved, commit the approved spec and plan separately, then verify the index is clear. If signing fails, stop and ask before starting implementation; do not change the index or attempt an unsigned commit.

## Review Focus

- Seed `0` and `4294967295`: both must be deterministic and valid, with zero-seed normalization identical in both runtimes (Task 1 and Task 3 tests).
- Width at the supported minimum and maximum: generation must stay within coordinate limits and reject a corner radius unsafe for the chosen half-width (Task 2 tests).
- Close or concave corners: centerline and derived road boundaries must not self-intersect or overlap (Task 2 tests).
- Unknown templates in Python and JavaScript, plus malformed seeds/widths in both runtimes: fail with a clear error rather than silently returning an oval (Tasks 1–3 tests).
- API-connected versus browser-only mode: the same template and seed must produce matching centerline coordinates and recipe metadata (Task 3 parity test).

---

### Task 1: Versioned Python corner-profile generation

**Files:**
- Modify: `local_simulator/track_generator.py`
- Test: `tests/test_track_generator.py`

**Interfaces:**
- Consumes: `CustomTrackGeometry`, `CustomMapSpec`, `validate_custom_geometry()`.
- Produces: `TEMPLATES` including `technical`, `GENERATOR_VERSION = 2`, and the existing `generate_custom_map(map_id, design_seed, template, width, max_steps, frame_skip) -> CustomMapSpec` contract. New maps record `template`, `design_seed`, `generator_version`, `corner_count`, and `corner_sequence` in `generator` metadata.

- [ ] **Step 1: Write failing profile and reproducibility tests**

```python
def test_generation_v2_records_distinct_corner_profiles(self):
    from local_simulator.track_generator import TEMPLATES, generate_custom_map

    self.assertEqual(
        TEMPLATES,
        frozenset({"oval", "s_curve", "hairpin", "chicane", "technical"}),
    )
    profiles = {}
    expected_ranges = {
        "oval": (4, 4),
        "s_curve": (6, 8),
        "hairpin": (6, 9),
        "chicane": (7, 10),
        "technical": (9, 12),
    }
    for template in sorted(TEMPLATES):
        first = generate_custom_map(f"custom-track-{template}", 90421, template)
        second = generate_custom_map(f"custom-track-{template}", 90421, template)
        first_meta = dict(first.generator)
        second_meta = dict(second.generator)
        self.assertEqual(first, second)
        self.assertEqual(first_meta["generator_version"], 2)
        self.assertEqual(first_meta["corner_count"], len(first_meta["corner_sequence"]))
        self.assertGreaterEqual(first_meta["corner_count"], expected_ranges[template][0])
        self.assertLessEqual(first_meta["corner_count"], expected_ranges[template][1])
        sequence = first_meta["corner_sequence"]
        self.assertEqual(len(sequence), first_meta["corner_count"])
        self.assertTrue(all(token.split(":", 1)[0] in {"left", "right"} for token in sequence))
        if template == "s_curve":
            self.assertTrue(any(sequence[index].split(":", 1)[0] != sequence[index + 1].split(":", 1)[0] for index in range(len(sequence) - 1)))
        elif template == "hairpin":
            self.assertGreaterEqual(sum(token.endswith(":hairpin") for token in sequence), 2)
        elif template == "chicane":
            switches = sum(sequence[index].split(":", 1)[0] != sequence[index + 1].split(":", 1)[0] for index in range(len(sequence) - 1))
            self.assertGreaterEqual(switches, 4)
        elif template == "technical":
            self.assertGreaterEqual(len({token.split(":", 1)[1] for token in sequence}), 3)
        profiles[template] = tuple(first_meta["corner_sequence"])
    self.assertNotEqual(profiles["oval"], profiles["technical"])


def test_boundary_seeds_are_repeatable_and_technical_seeds_change_shape(self):
    from local_simulator.track_generator import generate_custom_map

    for seed in (0, 4294967295):
        first = generate_custom_map(f"custom-track-boundary-{seed}", seed, "technical")
        second = generate_custom_map(f"custom-track-boundary-{seed}", seed, "technical")
        self.assertEqual(first, second)

    for template in ("oval", "s_curve", "hairpin", "chicane", "technical"):
        geometries = {
            generate_custom_map(f"custom-track-{template}-{seed}", seed, template).geometry.centerline
            for seed in range(16)
        }
        with self.subTest(template=template):
            self.assertGreaterEqual(len(geometries), 8)


def test_sampled_centerline_respects_segment_limit(self):
    import math
    from local_simulator.track_generator import TEMPLATES, generate_custom_map

    for template in sorted(TEMPLATES):
        geometry = generate_custom_map(f"custom-track-spacing-{template}", 42, template).geometry
        for index, point in enumerate(geometry.centerline):
            following = geometry.centerline[(index + 1) % len(geometry.centerline)]
            self.assertLessEqual(math.dist(point, following), 4.00001)


def test_unknown_template_is_rejected(self):
    from local_simulator.track_generator import generate_custom_map

    with self.assertRaisesRegex(ValueError, "template"):
        generate_custom_map("custom-track-invalid", 1, "unknown")


def test_invalid_design_seeds_are_rejected(self):
    from local_simulator.track_generator import generate_custom_map

    for seed in (True, -1, 4294967296, 1.5):
        with self.subTest(seed=seed):
            with self.assertRaisesRegex(ValueError, "design_seed"):
                generate_custom_map("custom-track-invalid-seed", seed, "oval")
```

Add these methods inside the existing `TestTrackGenerator` class.

- [ ] **Step 2: Run the focused test and verify the expected failure**

Run: `D:/HAIC/haic-env/Scripts/python.exe -m unittest tests.test_track_generator.TestTrackGenerator.test_generation_v2_records_distinct_corner_profiles -v`

Expected: FAIL because `technical` and versioned corner-profile metadata are not implemented.

- [ ] **Step 3: Implement deterministic corner recipes and a rounded closed loop**

Keep the generation API unchanged. Add an internal PRNG using the same unsigned 32-bit recurrence in Python and JavaScript:

```python
class _TrackRng:
    def __init__(self, seed: int) -> None:
        self.state = seed & 0xFFFFFFFF
        if self.state == 0:
            self.state = 1

    def random(self) -> float:
        self.state = (1664525 * self.state + 1013904223) & 0xFFFFFFFF
        return self.state / 4294967296.0


CORNER_COUNT_RANGES = {
    "oval": (4, 4),
    "s_curve": (6, 8),
    "hairpin": (6, 9),
    "chicane": (7, 10),
    "technical": (9, 12),
}
```

For each template, use a seeded recipe with these corner-count ranges: `oval` 4; `s_curve` 6–8; `hairpin` 6–9 including at least two hairpin corners; `chicane` 7–10 including at least two left-right pairs; `technical` 9–12 with mixed radii. Encode each recipe entry as `left:<class>` or `right:<class>`, where `<class>` is `wide`, `medium`, `tight`, or `hairpin`. `oval` uses four wide turns; `s_curve` includes both directions and a direction switch; `hairpin` includes at least two `:hairpin` entries; `chicane` has at least four adjacent direction switches (two left-right pairs); `technical` uses all three of wide/medium/tight. Select counts and token order with the seeded PRNG, then select seeded straight lengths and radial anchor offsets according to the token classes. Construct the anchor polygon in strictly increasing polar-angle order, fillet each corner, and accept a candidate only if its measured turn/radius features satisfy the encoded recipe. Use bounded deterministic retries for candidates that fail profile or geometry checks. Sample each curve and straight at no more than 4 world units per centerline segment; quantize coordinates using the same symmetric half-away-from-zero rule to five decimal places in both runtimes, then validate. Derive `corner_sequence` from the recipe, not from the number of sample points.

Use `CORNER_COUNT_RANGES` as the authoritative range table. For each candidate, draw one positive gap weight `0.5 + rng.random()` per anchor and normalize the weights to `2π`. Project the resulting gap vector toward equal spacing by the largest common factor that keeps every gap within `20°..100°`; this preserves deterministic seeded variation while guaranteeing valid polar ordering without seed-dependent redraw exhaustion. Mirror that projection exactly in JavaScript. Scale the base ellipse by `max(1.0, width / 8.0)` and use a 4096-point ceiling so even width 100 can retain the four-world-unit sampling guarantee. At each anchor, derive a quadratic fillet trim from the class-specific target radius and the local deflection angle, capped at 45% of the shorter adjoining edge; sample a quadratic Bézier through the anchor. Allocate straight samples with `ceil(segment_length / 4.0)` intervals. On candidate acceptance, derive signed turn and local-radius measurements from the quantized polyline and enforce each template's recipe constraints before writing metadata. Validate `design_seed` and `width` once before retrying so malformed caller inputs fail immediately rather than consuming the candidate budget.

Keep the simple four-corner `oval` as the reference profile. For all other profiles, the seed must affect the corner recipe or local radii/straight lengths, not just global scale or rotation. Reject unknown templates with the existing `ValueError` behavior. Set generator metadata through `CustomMapSpec(generator=...)` without changing the map schema.

- [ ] **Step 4: Run generator tests and verify every existing template still works**

Run: `D:/HAIC/haic-env/Scripts/python.exe -m unittest tests.test_track_generator -v`

Expected: PASS, including the pre-existing same-seed fingerprint, different-seed, batch-ID, and self-intersection tests.

- [ ] **Step 5: Commit the Python generator task**

```powershell
git add local_simulator/track_generator.py tests/test_track_generator.py
git commit -m "feat: generate deterministic multi-corner layouts"
```

### Task 2: Width-aware geometry safety and Box2D smoke coverage

**Files:**
- Modify: `local_simulator/track_generator.py`
- Modify: `tests/test_track_generator.py`
- Modify: `tests/test_custom_environment.py`

**Interfaces:**
- Consumes: Task 1's versioned centerline and recipe metadata.
- Produces: `validate_custom_geometry(geometry) -> None` rejects too-tight turns and overlapping non-adjacent road edges; internal `_road_edges_intersect(geometry) -> bool` supports direct testing; generation uses a bounded deterministic retry policy and raises a clear `ValueError` after exhaustion.

- [ ] **Step 1: Write failing tight-radius and road-clearance tests**

```python
def test_rejects_turn_radius_smaller_than_track_half_width(self):
    import math
    from local_simulator.track_generator import validate_custom_geometry
    from local_simulator.track_model import CustomTrackGeometry

    points = tuple(
        (3.0 * math.cos(2 * math.pi * index / 48), 3.0 * math.sin(2 * math.pi * index / 48))
        for index in range(48)
    )
    with self.assertRaisesRegex(ValueError, "turn radius"):
        validate_custom_geometry(CustomTrackGeometry(centerline=points, width=8.0))


def test_generator_handles_width_limits_and_rejects_out_of_range_width(self):
    from local_simulator.track_generator import generate_custom_map

    for width in (0.5, 100.0):
        document = generate_custom_map("custom-track-width", 73, "technical", width=width)
        self.assertTrue(all(abs(value) <= 100000 for point in document.geometry.centerline for value in point))
    with self.assertRaisesRegex(ValueError, "width"):
        generate_custom_map("custom-track-invalid-width", 73, "technical", width=100.1)


def test_generated_road_boundaries_do_not_intersect(self):
    from local_simulator.track_generator import TEMPLATES, _road_edges_intersect, generate_custom_map

    for template in sorted(TEMPLATES):
        document = generate_custom_map(f"custom-track-clearance-{template}", 73, template)
        self.assertFalse(_road_edges_intersect(document.geometry))


def test_detects_overlapping_road_boundaries(self):
    from local_simulator.track_generator import _road_edges_intersect
    from local_simulator.track_model import CustomTrackGeometry

    narrow_loop = CustomTrackGeometry(
        centerline=(
            (-20.0, -3.0), (-10.0, -3.0), (0.0, -3.0), (10.0, -3.0),
            (20.0, -3.0), (23.0, 0.0), (20.0, 3.0), (10.0, 3.0),
            (0.0, 3.0), (-10.0, 3.0), (-20.0, 3.0), (-23.0, 0.0),
        ),
        width=8.0,
    )
    self.assertTrue(_road_edges_intersect(narrow_loop))


def test_generated_profiles_validate_for_a_fixed_seed_range(self):
    from local_simulator.track_generator import TEMPLATES, generate_custom_map, validate_custom_geometry

    for template in sorted(TEMPLATES):
        for seed in range(100):
            with self.subTest(template=template, seed=seed):
                document = generate_custom_map(f"custom-track-{template}-{seed}", seed, template)
                validate_custom_geometry(document.geometry)
```

Add these methods inside the existing `TestTrackGenerator` class.

- [ ] **Step 2: Run focused tests and verify they fail for the new radius case**

Run: `D:/HAIC/haic-env/Scripts/python.exe -m unittest tests.test_track_generator.TestTrackGenerator.test_rejects_turn_radius_smaller_than_track_half_width -v`

Expected: FAIL because the current validator has no width-aware curvature or road-edge clearance check.

- [ ] **Step 3: Implement width-aware validation and bounded generation retries**

Estimate local curvature from each consecutive centerline triple using the triangle circumradius. For side lengths `a`, `b`, `c` and absolute doubled area `cross`, compute `radius = a * b * c / (2 * cross)`; treat `cross <= 1e-9` as a straight segment with infinite radius. `CustomCarRacing` offsets each edge by `geometry.width`, so in the current contract this value is the road half-width; reject a finite radius below `geometry.width + 0.05 * geometry.width`. Derive both offset road boundaries from normalized bisector normals with a bounded miter length; reject intersections between non-adjacent segments on either boundary. Also reject centerline segment pairs at least `max(4 * geometry.width, 8)` world units apart along the lap if their minimum distance is less than the full road width (`2 * geometry.width`); the along-track exclusion avoids mistaking neighboring samples on one continuous bend for separate road sections. Keep the existing centerline self-intersection check.

```python
for attempt in range(32):
    attempt_seed = (design_seed + attempt * 0x9E3779B9) & 0xFFFFFFFF
    geometry = _build_geometry_candidate(template, attempt_seed, width)
    try:
        validate_custom_geometry(geometry)
    except ValueError as error:
        last_error = error
        continue
    return geometry
raise ValueError(f"could not generate a valid {template} track: {last_error}")
```

Refactor `_generate_geometry()` to own this bounded retry loop and have `_build_geometry_candidate()` construct one candidate without recursively validating it. Keep `design_seed` as the user-supplied value in map metadata; apply retry seeds only to candidate construction. Python and JavaScript must use the same 32 attempts, seed progression, validation thresholds, and first-valid-candidate rule. Do not return a simpler fallback silently.

- [ ] **Step 4: Verify the 500-map seed sweep and custom environment execution**

Add this test in `tests/test_custom_environment.py`:

```python
def test_every_generated_template_runs_in_custom_environment(self):
    from local_simulator.environment import create_environment, reset_environment
    from local_simulator.track_generator import TEMPLATES, generate_custom_map

    for template in sorted(TEMPLATES):
        document = generate_custom_map(f"custom-track-env-{template}", 73, template)
        environment, _raw = create_environment(document, render_mode=None)
        try:
            observation, _info = reset_environment(environment, document)
            self.assertEqual(observation.shape, (4, 84, 84))
            next_observation, *_ = environment.step(
                np.array([0.0, 1.0, 0.0], dtype=np.float32)
            )
            self.assertEqual(next_observation.shape, (4, 84, 84))
        finally:
            environment.close()
```

Add this method inside the existing `TestCustomEnvironment` class.

Run: `D:/HAIC/haic-env/Scripts/python.exe -m unittest tests.test_track_generator tests.test_custom_environment -v`

Expected: PASS for all five profiles, the 0–99 seed sweep, existing malformed-geometry tests, and Box2D smoke tests.

- [ ] **Step 5: Commit geometry safety**

```powershell
git add local_simulator/track_generator.py tests/test_track_generator.py tests/test_custom_environment.py
git commit -m "fix: validate multi-corner road clearance"
```

### Task 3: Browser template selection, summary, and generator parity

**Files:**
- Modify: `web_simulator/index.html`
- Modify: `web_simulator/app.js`
- Modify: `web_simulator/styles.css`
- Modify: `tests/test_web_simulator_assets.py`
- Modify: `tests/test_web_simulator_client.py`

**Interfaces:**
- Consumes: Task 1's template IDs, PRNG, recipe metadata, and Task 2's valid geometry constraints.
- Produces: `makeBrowserCustomMap(options)` returns schema-2 maps with matching centerline and generator metadata; the map panel displays a concise corner summary.

- [ ] **Step 1: Write failing UI and Python–JavaScript parity tests**

In the existing asset test, assert `self.assertIn('value="technical"', html)` and `self.assertIn('id="generated-track-summary"', html)`. Add `import json` at module scope in `tests/test_web_simulator_client.py`, then add a Node-backed parity test that compares each template at seeds `0`, `42`, and `4294967295` against Python `generate_custom_map()` output:

```python
@unittest.skipUnless(NODE, "Node.js is required for browser client tests")
def test_browser_fallback_matches_python_generator(self):
    from local_simulator.track_generator import TEMPLATES, generate_custom_map

    for template in sorted(TEMPLATES):
        for seed in (0, 42, 4294967295):
            with self.subTest(template=template, seed=seed):
                expected = generate_custom_map(f"custom-track-{template}", seed, template)
                options = {
                    "map_id": f"custom-track-{template}",
                    "design_seed": seed,
                    "template": template,
                    "width": 8,
                    "max_steps": 2000,
                    "frame_skip": 4,
                    "obstacles": [],
                }
                script = r'''const fs = require("fs");
const vm = require("vm");
const source = fs.readFileSync("web_simulator/app.js", "utf8");
const context = { window: {}, document: { addEventListener() {} } };
vm.createContext(context);
vm.runInContext(source, context);
const options = JSON.parse(process.argv[1]);
const map = context.window.HAICSimulator.makeBrowserCustomMap(options);
console.log(JSON.stringify({ centerline: map.geometry.centerline, generator: map.generator }));'''
                result = subprocess.run(
                    [NODE, "-e", script, json.dumps(options)],
                    cwd=Path.cwd(), capture_output=True, text=True, check=False,
                )
                self.assertEqual(result.returncode, 0, result.stderr or result.stdout)
                actual = json.loads(result.stdout)
                expected_metadata = json.loads(json.dumps(dict(expected.generator)))
                self.assertEqual(actual["generator"], expected_metadata)
                self.assertEqual(
                    actual["centerline"],
                    [list(point) for point in expected.geometry.centerline],
                )


@unittest.skipUnless(NODE, "Node.js is required for browser client tests")
def test_browser_fallback_rejects_unknown_template(self):
    script = r'''const fs = require("fs");
const vm = require("vm");
const source = fs.readFileSync("web_simulator/app.js", "utf8");
const context = { window: {}, document: { addEventListener() {} } };
vm.createContext(context);
vm.runInContext(source, context);
let rejected = false;
try {
  context.window.HAICSimulator.makeBrowserCustomMap({
    map_id: "custom-track-invalid", design_seed: 1, template: "unknown",
    width: 8, max_steps: 2000, frame_skip: 4, obstacles: []
  });
} catch (error) {
  rejected = /template/i.test(error.message);
}
if (!rejected) throw new Error("unknown template was not clearly rejected");'''
    result = subprocess.run(
        [NODE, "-e", script],
        cwd=Path.cwd(), capture_output=True, text=True, check=False,
    )
    self.assertEqual(result.returncode, 0, result.stderr or result.stdout)


@unittest.skipUnless(NODE, "Node.js is required for browser client tests")
def test_browser_fallback_rejects_invalid_seed_and_width(self):
    script = r'''const fs = require("fs");
const vm = require("vm");
const source = fs.readFileSync("web_simulator/app.js", "utf8");
const context = { window: {}, document: { addEventListener() {} } };
vm.createContext(context);
vm.runInContext(source, context);
const base = {
  map_id: "custom-track-invalid", design_seed: 1, template: "oval",
  width: 8, max_steps: 2000, frame_skip: 4, obstacles: []
};
for (const invalid of [
  { design_seed: -1 }, { design_seed: 4294967296 }, { design_seed: true }, { width: 100.1 }
]) {
  let rejected = false;
  try {
    context.window.HAICSimulator.makeBrowserCustomMap({ ...base, ...invalid });
  } catch (error) {
    rejected = /design_seed|width/i.test(error.message);
  }
  if (!rejected) throw new Error(`invalid input was accepted: ${JSON.stringify(invalid)}`);
}'''
    result = subprocess.run(
        [NODE, "-e", script],
        cwd=Path.cwd(), capture_output=True, text=True, check=False,
    )
    self.assertEqual(result.returncode, 0, result.stderr or result.stdout)


```

Add these methods inside the existing `TestWebSimulatorClient` class. The existing module already imports `Path`, `shutil`, `subprocess`, and `unittest`; the only new module import is `json`. The parity test invokes the real Node VM helper and compares the complete JSON-compatible metadata and every five-decimal coordinate exactly; it does not replace either runtime with a mock. Update the existing fallback test so it checks 12–4096 centerline points and a maximum four-world-unit segment length rather than expecting the old fixed count of 48.

- [ ] **Step 2: Run the focused tests and verify the expected failures**

Run: `D:/HAIC/haic-env/Scripts/python.exe -m unittest tests.test_web_simulator_assets tests.test_web_simulator_client -v`

Expected: FAIL because the new template is absent from the Python/browser generators and page, and the browser still uses the old radial-loop algorithm.

- [ ] **Step 3: Port the versioned recipe into the browser fallback**

Add a track-only PRNG using `Math.imul(state, 1664525)` and unsigned 32-bit addition; leave the existing `seededRandom()` used for obstacle positions unchanged. Port the same template ranges and recipe-token grammar, angle ordering, anchor offsets, fillet math, sampling interval, five-decimal quantization, retry progression, validation thresholds, and metadata fields. The browser fallback must pick the same first valid candidate as Python. Reject unknown templates instead of generating an oval.

```javascript
function createTrackRandom(seed) {
  let state = seed >>> 0;
  if (state === 0) state = 1;
  return () => {
    state = (Math.imul(state, 1664525) + 1013904223) >>> 0;
    return state / 4294967296;
  };
}
```

- [ ] **Step 4: Add the new template and visible corner summary**

Add `<option value="technical">복합 기술형</option>` to `web_simulator/index.html` and a `generated-track-summary` element near the map status. In `applyCustomMap(payload)`, set that element from validated generator metadata, for example `9개 코너 · 헤어핀 2 · 시케인 1`; for imported legacy maps without recipe metadata, show `직접 제작 맵` rather than an invented count.

- [ ] **Step 5: Run browser asset and cross-runtime tests**

Run: `node --check web_simulator/app.js`

Run: `D:/HAIC/haic-env/Scripts/python.exe -m unittest tests.test_web_simulator_assets tests.test_web_simulator_client -v`

Expected: PASS; Python and JavaScript emit exactly equal centerlines and JSON metadata for every template and boundary seed, unknown templates are rejected, and the UI exposes the new template and summary.

- [ ] **Step 6: Commit browser parity**

```powershell
git add web_simulator/index.html web_simulator/app.js tests/test_web_simulator_assets.py tests/test_web_simulator_client.py
git commit -m "feat: mirror multi-corner generation in browser"
```

The worktree already has staged Task 7 web changes in these paths. They must be resolved in the signed preflight commit before Task 1. Before staging or committing this task, inspect both `git diff --cached` and `git diff`; do not drop or rewrite any pre-existing staged changes.

### Task 4: CLI/API coverage, user documentation, and end-to-end verification

**Files:**
- Modify: `tests/test_local_simulator_api.py`
- Modify: `tests/test_local_simulator_cli.py`
- Modify: `README.md`
- Modify: `tests/test_web_simulator_assets.py`
- Modify: `web_simulator/index.html`
- Modify: `web_simulator/app.js`

**Interfaces:**
- Consumes: Task 1's added `technical` template and Task 3's summary/seed behavior.
- Produces: documented supported styles, API and CLI regression coverage, and verified integrated browser behavior.

- [ ] **Step 1: Add API and CLI tests for the new template**

Add these tests to the existing classes and use their existing `request()` helper and temporary artifact roots:

```python
def test_technical_map_generation(self):
    response = self.request(
        "POST", "/api/maps/generate",
        {"map_kind": "custom", "map_id": "custom-track-api-technical",
         "design_seed": 42, "template": "technical"},
    )
    self.assertEqual(response["schema_version"], 2)
    self.assertEqual(response["generator"]["template"], "technical")
    self.assertEqual(response["generator"]["generator_version"], 2)
    self.assertGreaterEqual(response["generator"]["corner_count"], 9)


def test_technical_custom_map_cli_writes_recipe_metadata(self):
    from local_simulator.map import main as map_main

    with tempfile.TemporaryDirectory() as root:
        path = Path(root) / "technical.json"
        result = map_main([
            "--kind", "custom", "--template", "technical",
            "--design-seed", "42", "--output", str(path),
        ])
        self.assertEqual(result, 0)
        payload = json.loads(path.read_text(encoding="utf-8"))
        self.assertEqual(payload["generator"]["template"], "technical")
        self.assertEqual(payload["generator"]["generator_version"], 2)
        self.assertGreaterEqual(payload["generator"]["corner_count"], 9)
```

- [ ] **Step 2: Run the API and CLI tests**

Run: `D:/HAIC/haic-env/Scripts/python.exe -m unittest tests.test_local_simulator_api tests.test_local_simulator_cli -v`

Expected: PASS without adding template-specific branching to the API or CLI; both should consume `TEMPLATES`/`generate_custom_map()`.

- [ ] **Step 3: Document the track families and reproducibility**

Add a `### 자체 트랙 템플릿` subsection near the local simulator instructions in `README.md`. Include the five template IDs and Korean names, explain that `--design-seed` selects a reproducible corner recipe, show a `technical` map-generation command writing to `D:\HAIC\maps\technical-42.json`, state that schema 2 stays unchanged, and clarify that obstacles are a separate optional layer and custom tracks are for generalization tests rather than official evaluation.

- [ ] **Step 4: Run full tests and static checks**

Run: `node --check web_simulator/app.js`

Run: `git diff --check`

Run: `D:/HAIC/haic-env/Scripts/python.exe -W ignore::DeprecationWarning -m unittest discover -s tests -v`

Expected: PASS with no new dependency, schema, official-physics, or legacy-log regressions.

- [ ] **Step 5: Verify the integrated page without saving test artifacts**

Use the integrated page served from the feature worktree at `http://127.0.0.1:8766/`; leave any unrelated existing server on port 8765 untouched. Confirm the initial summary says `공식 트랙`; change the selected template without generating and confirm current geometry stays unchanged; then generate `oval`, `s_curve`, `hairpin`, `chicane`, and `technical` at the same seed, confirming each new geometry and visible corner summary. Change the seed and confirm the profile changes. Do not press **맵 저장** during this check; this is a visual check only and should not add files to `D:\HAIC\maps`.

- [ ] **Step 6: Commit documentation and integration coverage**

```powershell
git add tests/test_local_simulator_api.py tests/test_local_simulator_cli.py README.md
git commit -m "docs: document multi-corner custom tracks"
```

## Plan Self-Review

- Acceptance trace: (1) same-input geometry/fingerprint is checked in Task 1; (2) 16 seeds per template produce at least 8 geometries plus recipe constraints in Task 1; (3) every template passes seeds 0–99 in Task 2; (4) Python/browser centerlines and metadata compare exactly in Task 3; (5) each template is reset and stepped in Box2D in Task 2; (6) the existing schema-2 map and schema-1 run compatibility test remains in the full suite in Task 4; (7) template selection, preview update timing, summaries, and distinct silhouettes are checked in Tasks 3–4.
- Placeholder scan: each code task includes executable test examples and exact run commands; no TBD/TODO steps remain.
- Type consistency: Python stores `corner_sequence` as a tuple of strings in generator metadata, serialized as a JSON array; browser maps use the matching JSON array and integer `corner_count`.
- Review focus: seed boundaries and malformed seeds are covered in Tasks 1 and 3; width limits and turn/road clearance in Task 2; invalid templates in Tasks 1 and 3; exact runtime parity in Task 3.

# Extreme Technical Track Generator Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Generate deterministic, visibly distinct extreme tracks with 12, 14, or 16 major corners, at least three rapid S sections, and safe near-right-angle turns.

**Architecture:** Add a constructive orthogonal loop builder for `extreme_technical`, using a seeded outer loop and corner-counted inset notches before rounding each vertex into a drivable centerline. Mirror the same PRNG, topology recipe, rounding, and metadata in the browser fallback; keep map and run schemas unchanged.

**Tech Stack:** Python standard library, existing Box2D custom environment, browser JavaScript, Node.js client tests, Python `unittest`.

**Spec:** `docs/superpowers/specs/2026-09-20-extreme-technical-track-design.md`

## Global Constraints

- Extreme track corner counts are `12`, `14`, or `16`.
- Each track has at least `3` non-overlapping S sections; each S section is two opposite-direction corners with `0.6–2.2 * width` of straight remaining between their rounded fillets (and therefore no more than `3 * width`).
- At least `3` measured corner turns are between `75` and `105` degrees in absolute value.
- Generated `width` remains in the existing `0.5–9.0` range.
- Adjacent centerline samples remain at most `4` world units apart.
- Preserve the existing `MAX_CENTERLINE_POINTS = 4096` limit.
- The deterministic PRNG is 32-bit, and Python and JavaScript use the same draw order, geometry order, and coordinate quantization.
- Generator metadata records version `5`; map schema `2`, run schema, current API/CLI contracts, saved map geometry, official track, physics, and obstacle behavior remain unchanged.
- A failed extreme candidate may not silently lower its corner count or switch templates; it returns a clear error after bounded attempts.
- Do not add dependencies or write verification artifacts to `D:\HAIC`; use temporary directories for tests.

## Review Focus

- Seeds `0`, `1`, and `4294967295` and seeds selecting each topology count must be deterministic and must preserve that count on retries (Task 1 tests).
- The shortest S connector is measured after both corner fillets; skeleton vertex spacing includes both tangent trims, so it must meet the existing minimum-radius and road-boundary checks without exceeding `3 * width` of remaining straight, especially at width `9.0` (Tasks 1 and 3 tests).
- Malformed seeds, unsupported widths, and unknown templates must fail clearly in Python and browser generation instead of yielding a fallback oval (Tasks 1 and 2 tests).
- Python and JavaScript must agree exactly on geometry, counts, and metadata for fixed seeds and width boundaries (Task 2 parity tests).
- A generated map with each supported corner count must load, reset, and step in Box2D; prior schema-2 maps and schema-1 logs must remain readable (Tasks 1 and 3 tests).

---

## File Structure

- `local_simulator/track_generator.py` — Python PRNG, extreme route grammar, corner fillets, validation, and generator metadata.
- `tests/test_track_generator.py` — route closure, topology counts, seeded reproducibility, corner measurements, width limits, and 0–99 seed sweep.
- `tests/test_custom_environment.py` — Box2D reset/step smoke test for representative extreme layouts.
- `web_simulator/app.js` — matching browser fallback generator and generated-track summary.
- `web_simulator/index.html` — existing extreme template option and summary element; preserve both controls.
- `tests/test_web_simulator_client.py` — Node-backed browser/Python geometry and metadata parity plus summary formatting.
- `tests/test_web_simulator_assets.py` — ensure the extreme option and visible summary remain present.
- `tests/test_local_simulator_api.py` and `tests/test_local_simulator_cli.py` — schema, version, and generator metadata contract coverage.
- `README.md` — document the extreme template’s exact guarantees and generator version.

### Task 1: Build and validate the Python extreme route

**Files:**
- Modify: `local_simulator/track_generator.py`
- Modify: `tests/test_track_generator.py`
- Modify: `tests/test_custom_environment.py`

**Interfaces:**
- Consumes: `CustomTrackGeometry`, `CustomMapSpec`, `_TrackRng`, `validate_custom_geometry()`, and the existing `generate_custom_map(map_id, design_seed, template, width, max_steps, frame_skip) -> CustomMapSpec` contract.
- Produces: `_ExtremeRoute(vertices, corner_sequence, s_section_pairs)` and `_ExtremeGeometryResult(geometry, corner_sequence, corner_turn_degrees, corner_radius_widths, s_section_pairs, s_section_connector_lengths, near_90_corner_count)`.
- Produces: `_build_extreme_route(design_seed: int, width: float, attempt_index: int) -> _ExtremeRoute`, `_round_extreme_route(route: _ExtremeRoute, width: float) -> _ExtremeGeometryResult`, `_validate_extreme_profile(candidate: _ExtremeGeometryResult) -> None`, and `_generate_extreme_geometry(design_seed: int, width: float) -> _ExtremeGeometryResult`. `generate_custom_map()` dispatches to the new builder only for `extreme_technical`.

- [ ] **Step 1: Replace the current failing radial-profile test with a closed-topology test**

Replace `test_extreme_technical_template_generates_rapid_right_angle_s_sections` in
`tests/test_track_generator.py` with this test. It directly verifies closure, orthogonality,
disjoint S pairs, count diversity, and count-preserving retries:

```python
def test_extreme_route_uses_closed_orthogonal_skeleton(self):
    import math
    from local_simulator.track_generator import _build_extreme_route

    observed_counts = set()
    for width in (8.0, 9.0):
        for seed in (0, 1, 42, 73, 300, 1200, 4294967295):
            route = _build_extreme_route(seed, width, attempt_index=0)
            repeated_route = _build_extreme_route(seed, width, attempt_index=0)
            retry_route = _build_extreme_route(seed, width, attempt_index=1)
            count = len(route.vertices)
            observed_counts.add(count)
            self.assertEqual(route, repeated_route)
            self.assertIn(count, {12, 14, 16})
            self.assertEqual(len(route.corner_sequence), count)
            self.assertTrue(all(token.endswith(":tight") for token in route.corner_sequence))
            self.assertEqual(len(retry_route.vertices), count)
            self.assertEqual(len(retry_route.s_section_pairs), len(route.s_section_pairs))

            for index, point in enumerate(route.vertices):
                following = route.vertices[(index + 1) % count]
                dx = following[0] - point[0]
                dy = following[1] - point[1]
                self.assertGreater(math.hypot(dx, dy), 0.0)
                self.assertTrue(abs(dx) < 1e-7 or abs(dy) < 1e-7)

            directions = [token.split(":", 1)[0] for token in route.corner_sequence]
            used_corners = set()
            self.assertGreaterEqual(len(route.s_section_pairs), 3)
            for first, second in route.s_section_pairs:
                self.assertEqual(second, (first + 1) % count)
                self.assertNotIn(first, used_corners)
                self.assertNotIn(second, used_corners)
                self.assertNotEqual(directions[first], directions[second])
                radius_target = 1.7 * width * (1.0 + 0.4 * max(0.0, min(1.0, (width - 8.0) / 92.0)))
                tangent_trim = radius_target * math.sqrt(2.0)
                remaining_straight = math.dist(route.vertices[first], route.vertices[second]) - 2.0 * tangent_trim
                self.assertGreaterEqual(remaining_straight, 0.6 * width - 1e-5)
                self.assertLessEqual(remaining_straight, 2.2 * width + 1e-5)
                used_corners.update((first, second))

    self.assertEqual(observed_counts, {12, 14, 16})
```

- [ ] **Step 2: Run the new test and confirm the route-builder gap**

Run: `python -m unittest tests.test_track_generator.TestTrackGenerator.test_extreme_route_uses_closed_orthogonal_skeleton -v`

Expected: FAIL because `_build_extreme_route` does not yet exist. Do not relax the test to accept 12 corners for every seed.

- [ ] **Step 3: Implement the seeded closed-loop route grammar**

Add the internal immutable route type and helper signature:

```python
from dataclasses import dataclass


@dataclass(frozen=True)
class _ExtremeRoute:
    vertices: tuple[tuple[float, float], ...]
    corner_sequence: tuple[str, ...]
    s_section_pairs: tuple[tuple[int, int], ...]


@dataclass(frozen=True)
class _ExtremeGeometryResult:
    geometry: CustomTrackGeometry
    corner_sequence: tuple[str, ...]
    corner_turn_degrees: tuple[float, ...]
    corner_radius_widths: tuple[float, ...]
    s_section_pairs: tuple[tuple[int, int], ...]
    s_section_connector_lengths: tuple[float, ...]
    near_90_corner_count: int
```

Construct a simple orthogonal loop rather than choosing independent random turns. Orient the base rectangle counterclockwise so its interior is always on the left; its four convex turns are therefore `left`. Set its half-extents from the current width-aware scale (`150 + (width - 8) ± 6` on x and `93.75 + 0.625 * (width - 8) ± 4` on y). Select the target count once using the existing extreme count draw, `len(_corner_sequence("extreme_technical", _TrackRng(design_seed)))`; do not draw it from the per-attempt random stream. Initialize that stream with `(design_seed + attempt_index * 0x9E3779B9) & 0xFFFFFFFF`, matching the generator’s existing retry progression. For every candidate, use two inward rectangular notches for 12 corners, those same two notches plus one three-turn dogleg for 14, or three notches for 16. A notch replaces a straight side interval with the ordered turns `left, right, right, left`; its two disjoint S pairs are the first two and last two turns. Let `r = 1.7 * width * (1 + 0.4 * clamp((width - 8) / 92, 0, 1))`, the existing `tight` target radius, and `t = r * sqrt(2)`, the tangent trim of each 90-degree quadratic fillet. For each notch’s inset-depth edge, choose a remaining S-connector straight `g_s` in `0.6–2.2 * width` and set the skeleton edge length to `2*t + g_s`. For the connecting edge between the two parallel legs, choose `g_p` in `0.6–1.4 * width` and set that skeleton edge to `2*t + g_p`; this fits the shortest side at width `9.0` while leaving room for both fillets. Place no more than one notch on any side and keep its interval at least `6 * width` from both outer corners. For 12 corners, choose an opposing pair of sides. For 14 corners, put the two notches on the sides not incident to the selected dogleg corner and place both intervals toward the far ends from their shared opposite corner. For 16 corners, use both long sides and one seeded short side; place the long-side notches away from the selected short side. These placement rules keep adjacent notches apart even at maximum width. Replace the selected outer `left` corner at `C` with a `left-right-left` dogleg: if `u` is the incoming unit direction and `v` the outgoing unit direction, enter at `C - a*u`, then travel `b*v`, `a*u`, and rejoin the original outgoing edge at `C + b*v`. Choose `a` and `b` independently as `2*t + g_d` for `g_d` in `0.6–2.2 * width`; this adds two corners without changing the loop’s net turn and leaves a short, positive straight between each pair of dogleg fillets. Seed the selected sides, interval positions, gaps, rectangle proportions, reflection, and starting side. Sort insertions by perimeter position before constructing the cyclic vertex list. Build `corner_sequence` from the actual signed turns in vertex order and assign every skeleton corner the `:tight` class. In `_round_extreme_route()`, reuse `_quadratic_bezier()`, `_interpolate()`, `_round_coordinate()`, and `_measure_corner_profiles()`: tangent points are trimmed by `radius_target * sin(half_deflection) / cos(half_deflection)^2`, capped at 45% of each adjacent segment; use `max(4, ceil(2 * control_length / 4))` samples for a corner and `max(1, ceil(segment_length / 4))` for each straight, and omit the duplicate closing point. Compute each S connector’s actual remaining straight as its skeleton edge length minus the outgoing and incoming tangent trims of the two paired corners. Store these values in `s_section_connector_lengths`; require every value to be positive and at most `3 * width`, then quantize with `_round_coordinate()` and measure turns/radii with `_measure_corner_profiles()`. The final road-boundary check remains the acceptance authority. The seeded target-count draw is fixed from the original seed; `attempt_index` changes only seeded dimensions and placements, and every retry must preserve the target count and S-pair count.

Run: `python -m unittest tests.test_track_generator.TestTrackGenerator.test_extreme_route_uses_closed_orthogonal_skeleton -v`

Expected: PASS with all three corner counts represented and every returned route closed, orthogonal, and carrying at least three disjoint S pairs.

- [ ] **Step 4: Add failing tests for rounded geometry metadata and deterministic retries**

Add this method to `TestTrackGenerator`:

```python
def test_extreme_generation_reports_measured_corner_metadata(self):
    from local_simulator.track_generator import _generate_extreme_geometry, generate_custom_map

    observed_counts = set()
    for width in (8.0, 9.0):
        for seed in (0, 1, 42, 73, 300, 1200, 4294967295):
            candidate = _generate_extreme_geometry(seed, width)
            self.assertGreaterEqual(len(candidate.s_section_connector_lengths), 3)
            self.assertTrue(
                all(
                    0.6 * width - 1e-5 <= length <= 2.2 * width + 1e-5
                    for length in candidate.s_section_connector_lengths
                )
            )
            document = generate_custom_map(
                f"custom-track-extreme-{seed}", seed, "extreme_technical", width=width
            )
            metadata = dict(document.generator)
            count = metadata["corner_count"]
            observed_counts.add(count)
            self.assertIn(count, {12, 14, 16})
            self.assertEqual(len(metadata["corner_sequence"]), count)
            self.assertEqual(len(metadata["corner_turn_degrees"]), count)
            self.assertEqual(len(metadata["corner_radius_widths"]), count)
            self.assertGreaterEqual(metadata["s_section_count"], 3)
            measured_near_90 = sum(
                75.0 <= abs(turn) <= 105.0
                for turn in metadata["corner_turn_degrees"]
            )
            self.assertEqual(metadata["near_90_corner_count"], measured_near_90)
            self.assertGreaterEqual(measured_near_90, 3)

    self.assertEqual(observed_counts, {12, 14, 16})
    repeated = generate_custom_map("custom-track-extreme-repeat", 73, "extreme_technical")
    repeated_again = generate_custom_map("custom-track-extreme-repeat", 73, "extreme_technical")
    self.assertEqual(repeated, repeated_again)
    geometries = {
        generate_custom_map(
            f"custom-track-extreme-diversity-{seed}", seed, "extreme_technical"
        ).geometry.centerline
        for seed in range(16)
    }
    self.assertGreaterEqual(len(geometries), 8)
```

- [ ] **Step 5: Run the metadata test and verify it fails before implementation**

Run: `python -m unittest tests.test_track_generator.TestTrackGenerator.test_extreme_generation_reports_measured_corner_metadata -v`

Expected: FAIL because the current radial-anchor generator does not produce `s_section_count` and `near_90_corner_count` with a stable 12/14/16 layout.

- [ ] **Step 6: Round the route, measure actual corners, and wire the public generator**

Implement `_round_extreme_route()` with `_quadratic_bezier()`, `_interpolate()`, `_round_coordinate()`, and `_measure_corner_profiles()` as specified above. Round each 90-degree skeleton vertex with the same tangent-aligned quadratic fillet used by `_build_geometry_candidate()`, sample every segment at no more than four world units, and pass the resulting `CustomTrackGeometry` through `validate_custom_geometry()`. `_generate_extreme_geometry()` makes a bounded deterministic attempt sequence; every retry keeps the count and S-pair count chosen by the original seed. Reject candidates unless the measured turn/radius arrays each have one value per recipe corner, every corner’s measured sign matches the recipe, at least three turns are in `75–105` degrees, and the measured radii pass the existing corner-class limits. After exhaustion, raise `ValueError` with template, seed, width, and the last validation reason.

Use this control flow so candidate retries cannot silently choose a new corner count:

```python
def _generate_extreme_geometry(design_seed: int, width: float) -> _ExtremeGeometryResult:
    design_seed = _require_int("design_seed", design_seed, 0, MAX_SEED)
    width = _require_float("width", width, 0.5, MAX_GENERATED_TRACK_WIDTH)
    last_error = None
    for attempt_index in range(64):
        try:
            route = _build_extreme_route(design_seed, width, attempt_index)
            candidate = _round_extreme_route(route, width)
            _validate_extreme_profile(candidate)
            validate_custom_geometry(candidate.geometry)
        except ValueError as error:
            last_error = error
            continue
        return candidate
    raise ValueError(
        f"could not generate a valid extreme_technical track for seed {design_seed} "
        f"and width {width}: {last_error}"
    )
```

Implement `_validate_extreme_profile()` with the exact acceptance checks below:

```python
def _validate_extreme_profile(candidate: _ExtremeGeometryResult) -> None:
    count = len(candidate.corner_sequence)
    if count not in {12, 14, 16}:
        raise ValueError(f"extreme corner count {count} is unsupported")
    if len(candidate.s_section_pairs) < 3:
        raise ValueError("extreme route needs at least three S sections")
    if not (
        len(candidate.corner_turn_degrees)
        == len(candidate.corner_radius_widths)
        == count
    ):
        raise ValueError("extreme corner metadata lengths do not match the route")
    if len(candidate.s_section_connector_lengths) != len(candidate.s_section_pairs):
        raise ValueError("extreme S-section metadata lengths do not match the route")
    if any(
        not 0.6 * candidate.geometry.width - 1e-5
        <= length
        <= 2.2 * candidate.geometry.width + 1e-5
        for length in candidate.s_section_connector_lengths
    ):
        raise ValueError("extreme S-section connector must be 0.6–2.2 track widths")

    for token, turn, radius_widths in zip(
        candidate.corner_sequence,
        candidate.corner_turn_degrees,
        candidate.corner_radius_widths,
    ):
        direction, corner_class = token.split(":", 1)
        expected_sign = 1.0 if direction == "left" else -1.0
        if expected_sign * turn < 20.0:
            raise ValueError(f"measured corner turn does not match {token}")
        minimum_radius, maximum_radius = CORNER_RADIUS_WIDTH_RANGES[corner_class]
        if not minimum_radius <= radius_widths <= maximum_radius:
            raise ValueError(f"measured corner radius does not match {token}")

    near_90_count = sum(75.0 <= abs(turn) <= 105.0 for turn in candidate.corner_turn_degrees)
    if near_90_count < 3 or near_90_count != candidate.near_90_corner_count:
        raise ValueError("extreme route needs at least three measured near-90-degree corners")
```

Update `generate_custom_map()` to use the new path only for `extreme_technical`; leave the other five templates on `_generate_geometry()`. Set `GENERATOR_VERSION = 5`. For extreme maps, write `s_section_count = len(s_section_pairs)` and the measured `near_90_corner_count` in addition to the existing corner metadata. For other templates, retain their existing metadata fields and generation behavior. Rename `test_generation_v4_records_distinct_corner_profiles` to `test_generation_v5_records_distinct_corner_profiles`, update its expected version to 5, and additionally assert that its extreme template count is exactly in `{12, 14, 16}`. Update `test_generator_retries_enough_candidates_at_maximum_supported_width` to expect version 5.

- [ ] **Step 7: Verify rounded geometry, width edges, and the full seed sweep**

Extend `test_generator_handles_width_limits_and_rejects_out_of_range_width` to generate `extreme_technical` at `0.5` and `9.0` using seeds `0`, `42`, and `73` (covering 12, 14, and 16 corners at both bounds), then reject `9.01`. Extend `test_generated_profiles_validate_for_a_fixed_seed_range` so `extreme_technical` seeds `0–99` at width `8.0` all pass `validate_custom_geometry()`, have exactly 12/14/16 corners, report at least three S sections and three measured near-90 turns, and have `near_90_corner_count` equal to the measured turn count. Run:

`python -m unittest tests.test_track_generator -v`

Expected: PASS, including the pre-existing test that currently exposes the count-collapse bug, deterministic fingerprints, four-unit sample limit, road-boundary checks, width limits, and the new extreme seed sweep.

- [ ] **Step 8: Smoke-test Box2D for representative corner counts**

Add this method to `tests/test_custom_environment.py`:

```python
def test_extreme_corner_count_variants_reset_and_step(self):
    import numpy as np
    from local_simulator.environment import create_environment, reset_environment
    from local_simulator.track_generator import generate_custom_map

    observed_counts = set()
    for seed in (0, 42, 73):
        document = generate_custom_map(f"custom-track-env-{seed}", seed, "extreme_technical")
        observed_counts.add(dict(document.generator)["corner_count"])
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

    self.assertEqual(observed_counts, {12, 14, 16})
```

Run: `python -m unittest tests.test_custom_environment.TestCustomEnvironment.test_extreme_corner_count_variants_reset_and_step -v`

Expected: PASS for reset and one Box2D step for each supported count.

- [ ] **Step 9: Commit the Python generator and geometry coverage**

```powershell
git add local_simulator/track_generator.py tests/test_track_generator.py tests/test_custom_environment.py
git commit -m "feat: build extreme orthogonal track layouts"
```

### Task 2: Mirror the geometry in the browser and expose its summary

**Files:**
- Modify: `web_simulator/app.js`
- Retain and stage: `web_simulator/index.html` already has the extreme option and summary element
- Modify: `tests/test_web_simulator_client.py`
- Modify: `tests/test_web_simulator_assets.py`

**Interfaces:**
- Consumes: Python’s seeded route recipe and `_ExtremeGeometryResult` metadata contract from Task 1.
- Produces: `buildBrowserExtremeRoute(seed, width, attemptIndex) -> { vertices, cornerSequence, sSectionPairs }`, `roundBrowserExtremeRoute(route, width) -> { centerline, sequence, turns, radii, sSectionPairs, sSectionConnectorLengths, near90CornerCount }`, `validateBrowserExtremeProfile(candidate) -> void`, and `generateBrowserExtremeGeometry(seed, width)`.
- Produces: `window.HAICSimulator.makeBrowserCustomMap(options)` with schema-2 geometry and identical generator metadata; `formatGeneratedTrackSummary(map) -> string` for official, legacy custom, and extreme custom maps.

- [ ] **Step 1: Expand the parity test before changing the browser generator**

Update `test_browser_fallback_matches_python_generator` so `extreme_technical` uses seeds `0`, `1`, `42`, `73`, `300`, `1200`, and `4294967295`; keep seeds `0`, `42`, and `4294967295` for the other templates. Compare serialized generator metadata and every centerline point. Extend `test_browser_fallback_matches_python_at_width_limits` to compare `extreme_technical` at widths `0.5` and `9.0` using seeds `0`, `42`, and `73` so all three corner counts are exercised at each boundary; compare metadata and centerline there as well.

Run: `python -m unittest tests.test_web_simulator_client.TestWebSimulatorClient.test_browser_fallback_matches_python_generator -v`

Expected: FAIL because Python will now emit generator version 5 and the extreme metadata while the browser still implements the earlier radial recipe.

- [ ] **Step 2: Port the route grammar and fillets exactly to JavaScript**

Add JavaScript counterparts for `_build_extreme_route()`, `_round_extreme_route()`, and `_generate_extreme_geometry()` inside `web_simulator/app.js`. Use unsigned 32-bit arithmetic (`>>> 0`) and the same target-count draw, retry seed progression, topology construction order, notch placement order, fillet trims, sample allocation, and five-decimal quantization as Python. Keep the existing JavaScript generator for all other templates. Change `generateBrowserGeometry(template, designSeed, width)` to call `generateBrowserExtremeGeometry(designSeed, width)` immediately when the template is `extreme_technical`, before its existing retry loop; do not let extreme tracks pass through the radial-anchor candidate builder. In `makeBrowserCustomMap()`, set `generator_version: 5` and include the two extreme metrics only when the selected template is `extreme_technical`.

Use this retry structure and the exact helper return fields declared above:

```javascript
function generateBrowserExtremeGeometry(designSeed, width) {
  let lastError;
  for (let attemptIndex = 0; attemptIndex < 64; attemptIndex += 1) {
    try {
      const route = buildBrowserExtremeRoute(designSeed, width, attemptIndex);
      const candidate = roundBrowserExtremeRoute(route, width);
      validateBrowserExtremeProfile(candidate);
      validateBrowserCustomGeometry(candidate.centerline, width);
      return candidate;
    } catch (error) {
      lastError = error;
    }
  }
  throw new Error(
    `could not generate a valid extreme_technical track for seed ${designSeed} and width ${width}: ${lastError.message}`
  );
}
```

- [ ] **Step 3: Re-run the exact cross-runtime parity test**

Run:

`python -m unittest tests.test_web_simulator_client.TestWebSimulatorClient.test_browser_fallback_matches_python_generator tests.test_web_simulator_client.TestWebSimulatorClient.test_browser_fallback_matches_python_at_width_limits -v`

Expected: PASS for the listed seeds and width boundaries, with exact metadata and point equality.

- [ ] **Step 4: Add a failing generated-summary test**

Extract the summary formatting into a pure `formatGeneratedTrackSummary(map) -> string` function and expose that function through the existing `window.HAICSimulator` test surface. Add this method to `TestWebSimulatorClient`:

```python
@unittest.skipUnless(NODE, "Node.js is required for browser client tests")
def test_extreme_track_summary_reports_generated_metrics(self):
    script = r'''const fs = require("fs");
const vm = require("vm");
const source = fs.readFileSync("web_simulator/app.js", "utf8");
const context = { window: {}, document: { addEventListener() {} } };
vm.createContext(context);
vm.runInContext(source, context);
const format = context.window.HAICSimulator.formatGeneratedTrackSummary;
const summary = format({
  map_kind: "custom",
  generator: {
    template: "extreme_technical",
    corner_count: 14,
    s_section_count: 3,
    near_90_corner_count: 5
  }
});
if (summary !== "14개 코너 · S자 3구간 · 90° 급코너 5개") {
  throw new Error(`unexpected summary: ${summary}`);
}
const official = format({ map_kind: "official" });
if (official !== "공식 트랙") throw new Error(`unexpected official summary: ${official}`);
const legacy = format({
  map_kind: "custom",
  generator: {
    corner_count: 4,
    corner_sequence: ["left:wide", "left:wide", "left:wide", "left:wide"]
  }
});
if (legacy !== "4개 코너 · 헤어핀 0 · 방향 전환 0회") {
  throw new Error(`unexpected legacy summary: ${legacy}`);
}
const version4Extreme = format({
  map_kind: "custom",
  generator: {
    template: "extreme_technical",
    corner_count: 12,
    corner_sequence: Array(12).fill("left:wide")
  }
});
if (version4Extreme !== "12개 코너 · 헤어핀 0 · 방향 전환 0회") {
  throw new Error(`unexpected version-4 summary: ${version4Extreme}`);
}'''
    result = subprocess.run(
        [NODE, "-e", script], cwd=Path.cwd(), capture_output=True, text=True,
        encoding="utf-8", check=False,
    )
    self.assertEqual(result.returncode, 0, result.stderr or result.stdout)
```

Run: `python -m unittest tests.test_web_simulator_client.TestWebSimulatorClient.test_extreme_track_summary_reports_generated_metrics -v`

Expected: FAIL until the pure formatter is implemented and the browser summary uses it.

- [ ] **Step 5: Implement summary formatting without changing legacy summaries**

For official maps, return `공식 트랙`. For newly generated extreme maps with valid integer `corner_count`, `s_section_count`, and `near_90_corner_count`, use the exact format asserted above. Older version-4 extreme maps do not have the new metrics, so fall back to the existing corner, hairpin, and direction-switch summary; never render `undefined` or force a map migration. For other generated maps, retain that same existing summary. Make `updateGeneratedTrackSummary()` set `textContent` to the formatter result; do not add HTML markup from metadata.

- [ ] **Step 6: Run browser client and asset tests**

Add asset assertions that `web_simulator/index.html` still contains `value="extreme_technical"` and `id="generated-track-summary"`, and that `app.js` displays `S자` and `90° 급코너` for the extreme template. Run:

`python -m unittest tests.test_web_simulator_client tests.test_web_simulator_assets -v`

Expected: PASS for Python/JavaScript parity, summary text, invalid input rejection, legacy maps, and required controls.

- [ ] **Step 7: Commit browser generation and summary coverage**

```powershell
git add web_simulator/app.js web_simulator/index.html tests/test_web_simulator_client.py tests/test_web_simulator_assets.py
git commit -m "feat: mirror extreme track generation in browser"
```

### Task 3: Lock the API/CLI contract, documentation, and integrated behavior

**Files:**
- Modify: `tests/test_local_simulator_api.py`
- Modify: `tests/test_local_simulator_cli.py`
- Modify: `README.md`

**Interfaces:**
- Consumes: Task 1’s `generate_custom_map()` and Task 2’s browser summary/parity behavior.
- Produces: verified API/CLI metadata at version 5 and user documentation that describes the same corner, S-section, and near-90 guarantees as the generator.

- [ ] **Step 1: Lock the API contract to the extreme metrics**

Update `test_extreme_technical_map_generation` in `tests/test_local_simulator_api.py`:

```python
self.assertEqual(response["schema_version"], 2)
self.assertEqual(response["generator"]["template"], "extreme_technical")
self.assertEqual(response["generator"]["generator_version"], 5)
self.assertIn(response["generator"]["corner_count"], {12, 14, 16})
self.assertGreaterEqual(response["generator"]["s_section_count"], 3)
self.assertGreaterEqual(response["generator"]["near_90_corner_count"], 3)
```

Run: `python -m unittest tests.test_local_simulator_api.TestLocalSimulatorApi.test_extreme_technical_map_generation -v`

Expected: PASS after Task 1. If it fails, fix the public metadata propagation rather than adding template-specific geometry to `api.py`.

Also change the existing `test_technical_map_generation` assertion from generator version 4 to 5; `GENERATOR_VERSION` is shared by all templates.

- [ ] **Step 2: Add an extreme CLI round-trip test using a temporary directory**

Add this method to `TestSimulatorCli`. It writes only inside a temporary directory:

```python
def test_extreme_technical_custom_map_cli_writes_recipe_metadata(self):
    from local_simulator.map import main as map_main

    with tempfile.TemporaryDirectory() as root:
        path = Path(root) / "extreme.json"
        result = map_main([
            "--kind", "custom",
            "--template", "extreme_technical",
            "--design-seed", "73",
            "--output", str(path),
        ])
        self.assertEqual(result, 0)
        payload = json.loads(path.read_text(encoding="utf-8"))
        self.assertEqual(payload["schema_version"], 2)
        self.assertEqual(payload["generator"]["generator_version"], 5)
        self.assertIn(payload["generator"]["corner_count"], {12, 14, 16})
        self.assertGreaterEqual(payload["generator"]["s_section_count"], 3)
        self.assertGreaterEqual(payload["generator"]["near_90_corner_count"], 3)
```

Run: `python -m unittest tests.test_local_simulator_cli.TestSimulatorCli.test_extreme_technical_custom_map_cli_writes_recipe_metadata -v`

Expected: PASS with the same metadata as the API and Python generator.

Update the existing `test_technical_custom_map_cli_writes_recipe_metadata` assertion from generator version 4 to 5 for the same shared-version reason.

- [ ] **Step 3: Update the template table and generator-version note in README**

In the existing `자체 트랙 템플릿` table, change the extreme row to state `12·14·16개 주요 코너`, `S자 구간 최소 3개`, and `90도에 가까운 급코너 최소 3개`. Update the following version paragraph from generator version 4 to version 5. Keep the warning that custom tracks are for local generalization testing and are not official evaluation tracks.

- [ ] **Step 4: Run the focused Python integration tests**

Run:

`python -m unittest tests.test_track_generator tests.test_custom_environment tests.test_local_simulator_api tests.test_local_simulator_cli -v`

Expected: PASS for geometry, Box2D reset/step, API schema, CLI JSON round-trip, and prior map/run compatibility tests.

- [ ] **Step 5: Verify the live feature-worktree page without saving artifacts**

Open `http://127.0.0.1:8766/`. Generate the extreme template with seeds `42`, `73`, and `300`; confirm the summary reports the generated corner count, at least three S sections, and at least three near-90 corners. Confirm changing the seed changes the visible track while reusing the same seed reproduces it. Verify the page shows a visible error rather than another track if generation is rejected. Do not press **맵 저장** and do not create run logs during this visual check.

- [ ] **Step 6: Run cross-runtime and full regression checks**

Run:

`node --check web_simulator/app.js`

`git diff --check`

`python -m unittest discover -s tests -v`

Expected: PASS, including all existing schema-2 map and schema-1 run-log tests, Python/JavaScript exact parity, and every extreme geometry seed sweep.

- [ ] **Step 7: Commit API/CLI coverage and documentation**

```powershell
git add tests/test_local_simulator_api.py tests/test_local_simulator_cli.py README.md
git commit -m "docs: document extreme technical track guarantees"
```

## Plan Self-Review

- **Spec coverage:** The seeded 12/14/16 topology, disjoint S sections, near-90 measured turns, rounded geometry, bounded errors, and width limits are tested in Task 1. Box2D execution and the existing map/run contracts are covered in Tasks 1 and 3. Browser determinism, exact metadata, and the visible summary are covered in Task 2. API/CLI and README parity are covered in Task 3.
- **Implementation completeness:** Every task has concrete files, test names, commands, expected results, and code/interface details. No deferred or unspecified implementation steps remain.
- **Type consistency:** `_ExtremeRoute` carries loop vertices, one recipe token per vertex, and indexed disjoint S pairs. `_ExtremeGeometryResult` carries the same recipe plus `CustomTrackGeometry`, measured turns/radii, actual remaining S-connector lengths, and near-90 count. Connector lengths are validation-only; Python map metadata and browser JSON both expose the same `s_section_count` and `near_90_corner_count` fields.
- **Review focus:** Seed boundaries and retry invariants are pinned in Task 1; minimum/maximum widths and close notches are pinned in Tasks 1 and 3; invalid inputs are covered by existing Python/Node tests in Tasks 1 and 2; parity and quantization are covered in Task 2; Box2D and legacy contracts are covered in Tasks 1 and 3.

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

Estimate local curvature from each consecutive centerline triple using the triangle circumradius. For side lengths `a`, `b`, `c` and absolute doubled area `cross`, compute `radius = a * b * c / (2 * cross)`; treat `cross <= 1e-9` as a straight segment with infinite radius. Reject a finite radius below `geometry.width / 2 + 0.05 * geometry.width`, the half-width plus a 5% clearance margin. Derive both offset road boundaries from normalized bisector normals with a bounded miter length; reject intersections between non-adjacent segments on either boundary, and reject centerline segment pairs separated by at least one complete intervening segment if their minimum distance is less than the road width. Keep the existing centerline self-intersection check.

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


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

Use `CORNER_COUNT_RANGES` as the authoritative range table. For each candidate, draw one positive gap weight `0.5 + rng.random()` per anchor and normalize the weights to `2π`. Project the resulting gap vector toward equal spacing by the largest common factor that keeps every gap within `20°..100°`; this preserves deterministic seeded variation while guaranteeing valid polar ordering without seed-dependent redraw exhaustion. Mirror that projection exactly in JavaScript. Scale the base ellipse by `max(1.0, width / 8.0)` and use a 4096-point ceiling so even width 100 can retain the four-world-unit sampling guarantee. For each anchor, trim adjacent straight edges by `min(fillet_radius, 0.35 * min(incoming_length, outgoing_length))` and sample a quadratic Bézier through the anchor. Allocate straight samples with `ceil(segment_length / 4.0)` intervals. On candidate acceptance, derive signed turn and local-radius measurements from the quantized polyline and enforce each template's recipe constraints before writing metadata. Validate `design_seed` and `width` once before retrying so malformed caller inputs fail immediately rather than consuming the candidate budget.

Keep the simple four-corner `oval` as the reference profile. For all other profiles, the seed must affect the corner recipe or local radii/straight lengths, not just global scale or rotation. Reject unknown templates with the existing `ValueError` behavior. Set generator metadata through `CustomMapSpec(generator=...)` without changing the map schema.

- [ ] **Step 4: Run generator tests and verify every existing template still works**

Run: `D:/HAIC/haic-env/Scripts/python.exe -m unittest tests.test_track_generator -v`

Expected: PASS, including the pre-existing same-seed fingerprint, different-seed, batch-ID, and self-intersection tests.

- [ ] **Step 5: Commit the Python generator task**

```powershell
git add local_simulator/track_generator.py tests/test_track_generator.py
git commit -m "feat: generate deterministic multi-corner layouts"
```


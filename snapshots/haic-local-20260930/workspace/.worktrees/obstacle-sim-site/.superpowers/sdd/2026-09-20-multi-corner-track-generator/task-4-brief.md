### Task 4: CLI/API coverage, user documentation, and end-to-end verification

**Files:**
- Modify: `tests/test_local_simulator_api.py`
- Modify: `tests/test_local_simulator_cli.py`
- Modify: `README.md`

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

Use the existing integrated page at `http://127.0.0.1:8765/`. Change the selected template without generating and confirm the current geometry stays unchanged; then generate `oval`, `s_curve`, `hairpin`, `chicane`, and `technical` at the same seed, confirming each new geometry and visible corner summary. Change the seed and confirm the profile changes. Do not press **맵 저장** during this check; this is a visual check only and should not add files to `D:\HAIC\maps`.

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

### Task 3: Browser template selection, summary, and generator parity

**Files:**
- Modify: `web_simulator/index.html`
- Modify: `web_simulator/app.js`
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


import json
from pathlib import Path
import shutil
import subprocess
import unittest


NODE = shutil.which("node")


class TestWebSimulatorClient(unittest.TestCase):
    @unittest.skipUnless(NODE, "Node.js is required for browser client tests")
    def test_browser_fallback_map_preserves_seed_and_custom_id(self):
        script = r'''
const fs = require("fs");
const vm = require("vm");
const source = fs.readFileSync("web_simulator/app.js", "utf8");
const context = { window: {}, document: { addEventListener() {} } };
vm.createContext(context);
vm.runInContext(source, context);
const createMap = context.window.HAICSimulator.makeBrowserCustomMap;
const options = {
  map_id: "custom-track-fallback",
  design_seed: 37,
  template: "chicane",
  width: 8,
  max_steps: 250,
  frame_skip: 2,
  obstacles: []
};
const map = createMap(options);
if (map.schema_version !== 2 || map.map_kind !== "custom") throw new Error("wrong map schema");
if (map.map_id !== "custom-track-fallback") throw new Error("map id was not preserved");
if (map.generator.design_seed !== 37) throw new Error("design seed was not preserved");
if (map.geometry.centerline.length < 12 || map.geometry.centerline.length > 4096) throw new Error("fallback track has the wrong point count");
for (let index = 0; index < map.geometry.centerline.length; index += 1) {
  const point = map.geometry.centerline[index];
  const next = map.geometry.centerline[(index + 1) % map.geometry.centerline.length];
  if (Math.hypot(next[0] - point[0], next[1] - point[1]) > 4.00001) {
    throw new Error("fallback track has a segment longer than four world units");
  }
}
if (map.max_steps !== 250 || map.frame_skip !== 2) throw new Error("common run settings were not preserved");
const repeated = createMap(options);
if (JSON.stringify(map.geometry.centerline) !== JSON.stringify(repeated.geometry.centerline)) {
  throw new Error("fallback generation is not deterministic");
}
'''
        result = subprocess.run(
            [NODE, "-e", script],
            cwd=Path.cwd(),
            capture_output=True,
            text=True,
            check=False,
        )
        self.assertEqual(result.returncode, 0, result.stderr or result.stdout)

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
                        cwd=Path.cwd(), capture_output=True, text=True, encoding="utf-8", check=False,
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
            cwd=Path.cwd(), capture_output=True, text=True, encoding="utf-8", check=False,
        )
        self.assertEqual(result.returncode, 0, result.stderr or result.stdout)

    @unittest.skipUnless(NODE, "Node.js is required for browser client tests")
    def test_browser_fallback_matches_python_at_width_limits(self):
        from local_simulator.track_generator import generate_custom_map

        for width in (0.5, 9.0):
            with self.subTest(width=width):
                expected = generate_custom_map("custom-track-width", 73, "technical", width=width)
                options = {
                    "map_id": "custom-track-width",
                    "design_seed": 73,
                    "template": "technical",
                    "width": width,
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
const map = context.window.HAICSimulator.makeBrowserCustomMap(JSON.parse(process.argv[1]));
console.log(JSON.stringify(map.geometry.centerline));'''
                result = subprocess.run(
                    [NODE, "-e", script, json.dumps(options)],
                    cwd=Path.cwd(), capture_output=True, text=True, encoding="utf-8", check=False,
                )
                self.assertEqual(result.returncode, 0, result.stderr or result.stdout)
                self.assertEqual(
                    json.loads(result.stdout),
                    [list(point) for point in expected.geometry.centerline],
                )

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
  { design_seed: -1 }, { design_seed: 4294967296 }, { design_seed: true }, { width: 9.1 }, { width: 100 }
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
            cwd=Path.cwd(), capture_output=True, text=True, encoding="utf-8", check=False,
        )
        self.assertEqual(result.returncode, 0, result.stderr or result.stdout)

    @unittest.skipUnless(NODE, "Node.js is required for browser client tests")
    def test_custom_map_form_rejects_blank_and_fractional_design_seeds(self):
        script = r'''const fs = require("fs");
const vm = require("vm");
const source = fs.readFileSync("web_simulator/app.js", "utf8");
const seed = process.argv[1];
const context = {
  window: {},
  document: {
    addEventListener() {},
    getElementById(id) { return { value: id === "design-seed" ? seed : "" }; }
  }
};
vm.createContext(context);
vm.runInContext(source, context);
context.window.HAICSimulator.generateCustomMap().then(() => {
  throw new Error("invalid design seed was accepted");
}).catch((error) => {
  if (!/seed/i.test(error.message)) throw error;
});'''
        for seed in ("", "1.5"):
            with self.subTest(seed=seed):
                result = subprocess.run(
                    [NODE, "-e", script, seed],
                    cwd=Path.cwd(),
                    capture_output=True,
                    text=True,
                    encoding="utf-8",
                    check=False,
                )
                self.assertEqual(result.returncode, 0, result.stderr or result.stdout)

    @unittest.skipUnless(NODE, "Node.js is required for browser client tests")
    def test_legacy_map_and_run_payloads_remain_loadable(self):
        script = r'''
const fs = require("fs");
const vm = require("vm");
const source = fs.readFileSync("web_simulator/app.js", "utf8");
const context = { window: {}, document: { addEventListener() {}, getElementById() { return null; } } };
vm.createContext(context);
vm.runInContext(source, context);
const legacyMap = context.window.HAICSimulator.normalizeMap({ track_id: 1, seed: 42 });
if (legacyMap.schema_version !== 2 || legacyMap.map_kind !== "official" || legacyMap.map_id !== "official-track-1-seed-42") {
  throw new Error("legacy official map was not normalized");
}
const loaded = context.window.HAICSimulator.loadRunLog({
  schema_version: 1,
  run: { map: { track_id: 1, seed: 42 } },
  track: { width: 6.67, points: [[0, 0, 0, 0], [0.5, 0, 10, 0]] },
  steps: [{ position: [0, 0] }],
  summary: {}
});
if (loaded.log.schema_version !== 1 || loaded.log.steps.length !== 1) throw new Error("legacy run log was not loaded");
'''
        result = subprocess.run(
            [NODE, "-e", script],
            cwd=Path.cwd(),
            capture_output=True,
            text=True,
            check=False,
        )
        self.assertEqual(result.returncode, 0, result.stderr or result.stdout)

    @unittest.skipUnless(NODE, "Node.js is required for browser client tests")
    def test_manual_key_set_maps_to_steer_gas_brake_order(self):
        script = r'''
const fs = require("fs");
const vm = require("vm");
const source = fs.readFileSync("web_simulator/app.js", "utf8");
const context = { window: {}, document: { addEventListener() {} } };
vm.createContext(context);
vm.runInContext(source, context);
const toAction = context.window.HAICSimulator.manualActionFromKeys;
if (JSON.stringify(toAction(new Set(["ArrowLeft", "ArrowUp", "Space"]))) !== "[-1,1,1]") {
  throw new Error("left, gas, and brake must map to [-1, 1, 1]");
}
if (JSON.stringify(toAction(new Set(["ArrowRight", "ArrowUp"]))) !== "[1,1,0]") {
  throw new Error("right and gas must map to [1, 1, 0]");
}
if (JSON.stringify(toAction(new Set(["ArrowLeft", "ArrowRight"]))) !== "[0,0,0]") {
  throw new Error("opposite steering keys must cancel");
}
if (JSON.stringify(toAction(new Set())) !== "[0,0,0]") throw new Error("released keys must return to zero");
'''
        result = subprocess.run(
            [NODE, "-e", script],
            cwd=Path.cwd(),
            capture_output=True,
            text=True,
            check=False,
        )
        self.assertEqual(result.returncode, 0, result.stderr or result.stdout)

    @unittest.skipUnless(NODE, "Node.js is required for browser client tests")
    def test_track_canvas_draws_the_live_manual_vehicle(self):
        script = r'''const fs = require("fs");
const vm = require("vm");
const source = fs.readFileSync("web_simulator/app.js", "utf8");
const context = { window: {}, document: { addEventListener() {}, getElementById() { return null; } } };
vm.createContext(context);
vm.runInContext(source, context);
const calls = [];
const fills = [];
const drawing = {
  clearRect() {}, fillRect() {}, beginPath() {}, lineTo() {}, moveTo() {}, closePath() {},
  fill() {}, stroke() {}, arc() {}, setLineDash() {}, save() { calls.push("save"); },
  translate() { calls.push("translate"); }, rotate() { calls.push("rotate"); },
  restore() { calls.push("restore"); }
};
Object.defineProperty(drawing, "fillStyle", { set(value) { fills.push(value); } });
const canvas = { width: 400, height: 300, getContext() { return drawing; } };
context.window.HAICSimulator.drawTrack(canvas, {
  track: { width: 8, points: [[0, 0, 0, 0], [0.25, 0, 10, 0], [0.5, 0, 10, 10], [0.75, 0, 0, 10]] },
  official_obstacles: []
}, [], null, { position: [5, 4], angle: 0.5 });
if (!fills.includes("#ff6f74") || !calls.includes("translate") || !calls.includes("rotate")) {
  throw new Error("live car marker was not drawn at its latest position and angle");
}'''
        result = subprocess.run(
            [NODE, "-e", script],
            cwd=Path.cwd(),
            capture_output=True,
            text=True,
            encoding="utf-8",
            check=False,
        )
        self.assertEqual(result.returncode, 0, result.stderr or result.stdout)


if __name__ == "__main__":
    unittest.main()

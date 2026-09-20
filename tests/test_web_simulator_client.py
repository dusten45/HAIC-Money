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
if (map.geometry.centerline.length !== 48) throw new Error("fallback track has the wrong point count");
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


if __name__ == "__main__":
    unittest.main()

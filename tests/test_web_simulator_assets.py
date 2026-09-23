from pathlib import Path
import unittest


class TestWebAssets(unittest.TestCase):
    def test_site_has_local_assets_and_required_controls(self):
        html = Path("web_simulator/index.html").read_text(encoding="utf-8")
        self.assertIn('id="map-canvas"', html)
        self.assertIn('id="play-button"', html)
        self.assertIn('id="log-file"', html)
        self.assertIn('src="app.js"', html)
        self.assertNotIn("http://", html)
        self.assertNotIn("https://", html)

    def test_site_has_playback_and_metrics_controls(self):
        html = Path("web_simulator/index.html").read_text(encoding="utf-8")
        self.assertIn('id="record-frames" type="checkbox" checked', html)
        for element_id in (
            "pause-button",
            "step-forward",
            "step-back",
            "speed",
            "run-table",
            "metric-action",
            "metric-speed",
            "agent-select",
            "agent-status",
        ):
            self.assertIn(f'id="{element_id}"', html)

    def test_map_renderer_has_default_empty_state_target(self):
        script = Path("web_simulator/app.js").read_text(encoding="utf-8")
        self.assertIn('emptyId = "canvas-empty"', script)

    def test_map_renderer_projects_track_world_coordinates(self):
        script = Path("web_simulator/app.js").read_text(encoding="utf-8")
        self.assertIn("project([point[2], point[3]], bounds", script)

    def test_site_contains_custom_generator_and_run_controls(self):
        html = Path("web_simulator/index.html").read_text(encoding="utf-8")
        for element_id in (
            "map-kind",
            "custom-template",
            "design-seed",
            "generate-custom-map",
            "start-agent-run",
            "start-manual-run",
            "finish-manual-run",
            "local-api-status",
        ):
            self.assertIn(f'id="{element_id}"', html)
        self.assertIn('value="technical"', html)
        self.assertIn('value="extreme_technical"', html)
        self.assertIn('id="generated-track-summary"', html)
        script = Path("web_simulator/app.js").read_text(encoding="utf-8")
        self.assertIn("S자", script)
        self.assertIn("90° 급코너", script)

    def test_initial_track_summary_matches_the_official_default_map(self):
        html = Path("web_simulator/index.html").read_text(encoding="utf-8")
        script = Path("web_simulator/app.js").read_text(encoding="utf-8")
        self.assertIn('id="generated-track-summary" class="hint generated-track-summary">공식 트랙</p>', html)
        initialize = script.split("function initialize()", 1)[1].split("window.HAICSimulator", 1)[0]
        self.assertIn("updateGeneratedTrackSummary(state.mapSpec)", initialize)

    def test_script_contains_api_fallback_and_manual_action_names(self):
        script = Path("web_simulator/app.js").read_text(encoding="utf-8")
        for token in (
            "/api/health",
            "/api/agents",
            "loadAgents",
            "agent_id",
            "generateCustomMap",
            "startRun",
            "sendManualAction",
            "custom-track",
        ):
            self.assertIn(token, script)


if __name__ == "__main__":
    unittest.main()

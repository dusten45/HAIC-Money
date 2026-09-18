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
        for element_id in ("pause-button", "step-forward", "step-back", "speed", "run-table"):
            self.assertIn(f'id="{element_id}"', html)

    def test_map_renderer_has_default_empty_state_target(self):
        script = Path("web_simulator/app.js").read_text(encoding="utf-8")
        self.assertIn('emptyId = "canvas-empty"', script)

    def test_map_renderer_projects_track_world_coordinates(self):
        script = Path("web_simulator/app.js").read_text(encoding="utf-8")
        self.assertIn("project([point[2], point[3]], bounds", script)


if __name__ == "__main__":
    unittest.main()
